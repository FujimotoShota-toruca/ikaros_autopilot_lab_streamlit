# IKAROS-GO! v12 (Full / deterministic)
# - 状態遷移行列（感度行列）っぽいもので B-plane を動かす
# - 発電（太陽）と通信（地球）を“姿勢（β_in/out）”から判定
# - 乱数なし：同じ操作なら同じ結果（再現性100%）
#
# 注意：本アプリは教育用の「模型」です（IKAROSの実運用を正確に再現するものではありません）

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, List, Dict

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from PIL import Image
import io
import pathlib


APP_BUILD = "v13.4-orbit-comm-power-2026-02-14"

AU_KM = 149_597_870.7  # 1 AU in km
DEFAULT_TEX_PATH = pathlib.Path(__file__).parent / "assets" / "ikaros_texture.png"


# -----------------------------
# 便利関数
# -----------------------------
def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def unit(v: np.ndarray) -> np.ndarray:
    n = norm(v)
    return v / (n + 1e-12)


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    aa, bb = unit(a), unit(b)
    c = float(np.clip(np.dot(aa, bb), -1.0, 1.0))
    return float(math.degrees(math.acos(c)))


# -----------------------------
# 設定（ゲームのルール）
# -----------------------------
@dataclass
class CFG:
    n_turns: int = 5

    # B-plane
    target_bt_br_km: Tuple[float, float] = (25000.0, -12000.0)   # 目標（中心）
    score_radius_km: float = 60000.0                             # 0点になる半径
    tolerance_km: float = 8000.0                                 # 許容誤差（円）
    venus_radius_km: float = 6052.0                              # 金星の半径（表示用）

    # 操作（β_in/out）
    beta_in_deg_lim: float = 25.0
    beta_out_deg_lim: float = 25.0

    # “本格っぽさ”：感度行列のスケール（km/deg）
    sens_scale: float = 420.0

    # β=0でもズレる（決定論的ドリフト）
    drift_per_turn_km: float = 600.0

    # 通信・発電ルール（姿勢制約）
    comm_cone_half_deg: float = 60.0      # 帆法線（アンテナ向き）中心の通信コーン半角
    sun_tilt_limit_deg: float = 45.0      # 太陽方向との角度がこれ以上だと「運用上しんどい」扱い

    # ブラックアウト（通信できない時間帯）の模型：日数で決め打ち
    total_days: float = 180.0
    blackout_windows_days: Tuple[Tuple[float, float], ...] = ((55.0, 65.0), (115.0, 125.0))


C = CFG()


# -----------------------------
# 軌道（2D/3Dの絵用：簡単な円軌道＋補間）
# -----------------------------
def get_positions_3d(day: float, total_days: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    太陽・地球・金星・IKAROS の“雰囲気”位置（3Dだけど z=0）
    - 地球/金星：円軌道（超ざっくり）
    - IKAROS：地球(出発)→金星(到着)を結ぶ “ベジェ曲線” の1本
      ※最後は必ず「金星の位置」と一致します（見た目を気持ちよくするため）
    単位は AU（天文単位）っぽい相対値。
    """
    t = day / max(total_days, 1e-9)
    t = float(np.clip(t, 0.0, 1.0))

    # 太陽
    sun = np.zeros(3)

    # 地球（半径1 AU）
    wE = 2 * math.pi / 365.25
    thE = wE * day + 0.2
    earth = np.array([math.cos(thE), math.sin(thE), 0.0], dtype=float)

    # 金星（半径0.723 AU）
    wV = 2 * math.pi / 224.7
    thV = wV * day - 0.8
    venus = 0.723 * np.array([math.cos(thV), math.sin(thV), 0.0], dtype=float)

    # --- IKAROSの“転移曲線”：出発(地球@day0)→到着(金星@day=total_days) ---
    thE0 = 0.2
    earth0 = np.array([math.cos(thE0), math.sin(thE0), 0.0], dtype=float)

    thVF = wV * total_days - 0.8
    venusF = 0.723 * np.array([math.cos(thVF), math.sin(thVF), 0.0], dtype=float)

    # 速度っぽい向き（円軌道の接線方向）
    tanE = unit(np.array([-earth0[1], earth0[0], 0.0], dtype=float))
    tanV = unit(np.array([-venusF[1], venusF[0], 0.0], dtype=float))

    # 制御点（曲がり具合）
    L = 0.55 * norm(earth0 - venusF)
    p0 = earth0
    p1 = earth0 + L * tanE
    p2 = venusF - L * tanV
    p3 = venusF

    # 3次ベジェ
    sc = ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * (t ** 2) * p2 + (t ** 3) * p3
    return sun, earth, venus, sc


def in_blackout(day: float) -> bool:
    for a, b in C.blackout_windows_days:
        if a <= day <= b:
            return True
    return False


# -----------------------------
# 姿勢（β_in/out）→ 帆法線ベクトル（3D）
# -----------------------------
def sail_normal(sc: np.ndarray, sun: np.ndarray, earth: np.ndarray, beta_in: float, beta_out: float) -> np.ndarray:
    """
    “本格っぽく見える”ための姿勢模型：
    - 基準は太陽方向（sun_dir）
    - beta_in/out で、その周りの直交基底を使って傾ける（小角近似っぽい）
    """
    sun_dir = unit(sun - sc)
    earth_dir = unit(earth - sc)

    # 直交基底：e1,e2 は sun_dir に垂直
    e1 = unit(np.cross(sun_dir, np.array([0.0, 0.0, 1.0])))
    if norm(e1) < 1e-6:
        e1 = unit(np.cross(sun_dir, np.array([0.0, 1.0, 0.0])))
    e2 = unit(np.cross(sun_dir, e1))

    # 地球方向に少し寄せたいので、e1/e2 の符号を地球方向に合わせる（見た目の安定）
    if np.dot(earth_dir, e1) < 0:
        e1 = -e1
    if np.dot(earth_dir, e2) < 0:
        e2 = -e2

    # 小角っぽく傾ける（deg→rad）
    bi = math.radians(beta_in)
    bo = math.radians(beta_out)
    n = sun_dir + bi * e1 + bo * e2
    return unit(n)


def power_percent(sail_n: np.ndarray, sun_dir: np.ndarray) -> float:
    """発電の模型：太陽方向に正面ほど大きい（0〜100%）"""
    return float(max(0.0, float(np.dot(unit(sail_n), unit(sun_dir)))) * 100.0)


def comm_ok(sail_n: np.ndarray, earth_dir: np.ndarray) -> bool:
    """
    通信の模型：帆法線（アンテナ向き）中心のコーン内に地球方向が入ればOK
    反対側（-sail_n）も同じようにOK扱い（アンテナが両面にあるイメージ）
    """
    ea = angle_deg(sail_n, earth_dir)
    ea2 = angle_deg(-sail_n, earth_dir)
    return (ea <= C.comm_cone_half_deg) or (ea2 <= C.comm_cone_half_deg)


def comm_strength_0_100(sail_n: np.ndarray, earth_dir: np.ndarray) -> float:
    """通信の“強さ”の模型（0〜100）：コーン中心に近いほど強い"""
    ea = min(angle_deg(sail_n, earth_dir), angle_deg(-sail_n, earth_dir))
    if ea >= C.comm_cone_half_deg:
        return 0.0
    return float(100.0 * (1.0 - ea / max(C.comm_cone_half_deg, 1e-9)))

# -----------------------------
# “物理っぽい”電力・通信（距離 + 指向性）
# -----------------------------
def power_percent_physical(sail_n: np.ndarray, sun_dir: np.ndarray, r_sun_au: float) -> float:
    """
    発電（相対％）の“物理っぽい”模型（簡易）
    - 入射角：cos(theta)（0以下は0）
    - 太陽距離：1/r^2（AU）
    100% =「1 AUで真正面（theta=0）」のとき
    """
    inc = max(0.0, float(np.dot(unit(sail_n), unit(sun_dir))))
    r = max(1e-6, float(r_sun_au))
    return float(min(100.0, 100.0 * inc / (r * r)))


def _gain_cos_n(angle_rad: float, half_power_deg: float) -> float:
    """
    cos^n パターンでアンテナ指向性を作る（簡易）
    half_power_deg で gain=0.5 になるよう n を決める。
    """
    a = float(angle_rad)
    if a >= math.pi / 2:
        return 0.0
    hp = math.radians(max(1e-3, float(half_power_deg)))
    c_hp = max(1e-6, math.cos(hp))
    n = math.log(0.5) / math.log(c_hp)
    return float(max(0.0, math.cos(a)) ** n)


def comm_rate_percent_physical(sail_n: np.ndarray, earth_dir: np.ndarray, r_earth_au: float, half_power_deg: float) -> float:
    """
    通信（相対％）の“物理っぽい”模型（簡易）
    - 距離減衰：1/r^2（AU）
    - 指向性：cos^n(角度)（half_power_deg で半減）
    100% = 「1 AUでコーン中心（角度0）」のとき
    """
    r = max(1e-6, float(r_earth_au))
    ang1 = math.radians(angle_deg(sail_n, earth_dir))
    ang2 = math.radians(angle_deg(-sail_n, earth_dir))
    ang = min(ang1, ang2)
    gain = _gain_cos_n(ang, half_power_deg)
    return float(min(100.0, 100.0 * gain / (r * r)))

# -----------------------------
# 状態遷移行列（感度行列）っぽいもの
# -----------------------------
def get_sensitivity(turn: int) -> np.ndarray:
    """
    2x2 感度行列 C(turn)（km/deg）
    本当のSTMではないけど、「この操作がB-planeにどう効くか」を表す模型。
    """
    f = 1.0 + 0.12 * math.sin(0.9 * (turn + 1))
    g = 1.0 + 0.10 * math.cos(0.7 * (turn + 1))

    base = np.array([[1.00, -0.30],
                     [0.55,  0.90]], dtype=float)
    M = C.sens_scale * np.array([[f, 0.0],
                                 [0.0, g]], dtype=float) @ base
    return M


def drift_vec(turn: int) -> np.ndarray:
    """β=0でもズレる（決定論的）"""
    a = 0.8 * (turn + 1) + 0.3
    b = 1.1 * (turn + 1) + 0.1
    return C.drift_per_turn_km * np.array([math.cos(a), math.sin(b)], dtype=float)


# -----------------------------
# ダーツ盤（B-plane）描画
# -----------------------------
def add_filled_ring(fig: go.Figure, center: np.ndarray, r_in: float, r_out: float) -> None:
    th = np.linspace(0.0, 2 * math.pi, 240)
    outer = np.c_[center[0] + r_out * np.cos(th), center[1] + r_out * np.sin(th)]
    inner = np.c_[center[0] + r_in * np.cos(th[::-1]), center[1] + r_in * np.sin(th[::-1])]
    poly = np.vstack([outer, inner, outer[:1]])
    fig.add_trace(go.Scatter(
        x=poly[:, 0], y=poly[:, 1],
        mode="lines",
        fill="toself",
        line=dict(width=1),
        opacity=0.18,
        showlegend=False,
    ))


def bplane_figure(
    x_hat: np.ndarray,
    x_true: np.ndarray,
    target: np.ndarray,
    pred_next: np.ndarray,
    pred_circle_r: float,
    history: np.ndarray,
) -> go.Figure:
    fig = go.Figure()

    rings = [(0.00, 0.15), (0.15, 0.30), (0.30, 0.50), (0.50, 0.75), (0.75, 1.00)]
    for a, b in rings:
        add_filled_ring(fig, target, a * C.score_radius_km, b * C.score_radius_km)

    th = np.linspace(0.0, 2 * math.pi, 240)

    # 金星の半径（原点中心）
    fig.add_trace(go.Scatter(
        x=0.0 + C.venus_radius_km * np.cos(th),
        y=0.0 + C.venus_radius_km * np.sin(th),
        mode="lines",
        name="金星の半径（表示）",
        line=dict(width=2),
    ))

    # 目標・許容誤差
    fig.add_trace(go.Scatter(
        x=[target[0]], y=[target[1]],
        mode="markers+text",
        name="目標",
        text=["★"],
        textposition="top center",
    ))
    fig.add_trace(go.Scatter(
        x=target[0] + C.tolerance_km * np.cos(th),
        y=target[1] + C.tolerance_km * np.sin(th),
        mode="lines",
        name="許容誤差（円）",
        line=dict(width=1, dash="dot"),
    ))

    # 履歴（推定）
    fig.add_trace(go.Scatter(
        x=history[:, 0], y=history[:, 1],
        mode="markers",
        name="これまで（推定）",
    ))

    # 現在（推定）
    fig.add_trace(go.Scatter(
        x=[x_hat[0]], y=[x_hat[1]],
        mode="markers",
        name="いまの位置（推定）",
    ))

    # 真値（参考）
    fig.add_trace(go.Scatter(
        x=[x_true[0]], y=[x_true[1]],
        mode="markers",
        name="真の位置（参考）",
        marker=dict(symbol="x"),
        opacity=0.5,
    ))

    # 予測矢印
    fig.add_trace(go.Scatter(
        x=[x_hat[0], pred_next[0]], y=[x_hat[1], pred_next[1]],
        mode="lines+markers",
        name="予測（次の平均）",
    ))

    # 予測の丸（目安）
    r = float(pred_circle_r)
    fig.add_trace(go.Scatter(
        x=pred_next[0] + r * np.cos(th),
        y=pred_next[1] + r * np.sin(th),
        mode="lines",
        name="予測のばらつき（目安）",
        line=dict(width=1, dash="dot"),
    ))

    lim = C.score_radius_km * 1.05
    fig.update_layout(
        xaxis_title="BT [km]",
        yaxis_title="BR [km]",
        height=720,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    fig.update_xaxes(range=[target[0] - lim, target[0] + lim], zeroline=False)
    fig.update_yaxes(range=[target[1] - lim, target[1] + lim], zeroline=False, scaleanchor="x", scaleratio=1)
    return fig


# -----------------------------
# ゲーム状態
# -----------------------------
def score_from_distance(d_km: float) -> int:
    s = 100.0 * max(0.0, 1.0 - d_km / (C.score_radius_km + 1e-12))
    return int(round(s))


def init_state() -> None:
    st.session_state.turn = 0
    st.session_state.day = 0.0

    # “真の効き”と“想定の効き”を固定（乱数なし）
    st.session_state.k_true = 1.15
    st.session_state.k_hat = 1.00

    st.session_state.x_true = np.array([0.0, 0.0], dtype=float)
    st.session_state.x_hat = np.array([0.0, 0.0], dtype=float)

    st.session_state.history = [st.session_state.x_hat.copy()]
    st.session_state.log: List[Dict[str, float]] = []
def step_once(beta_in: float, beta_out: float) -> None:
    turn = int(st.session_state.turn)
    if turn >= C.n_turns - 1:
        return

    u = np.array([beta_in, beta_out], dtype=float)

    # 真の遷移
    C_true = st.session_state.k_true * get_sensitivity(turn)
    d = drift_vec(turn)
    st.session_state.x_true = st.session_state.x_true + d + C_true @ u

    # 推定の遷移
    C_hat = st.session_state.k_hat * get_sensitivity(turn)
    st.session_state.x_hat = st.session_state.x_hat + d + C_hat @ u

    # 幾何（通信/発電判定）
    day = float(st.session_state.day)
    sun, earth, venus, sc = get_positions_3d(day, C.total_days)
    sun_dir = unit(sun - sc)
    earth_dir = unit(earth - sc)
    sail_n = sail_normal(sc, sun, earth, beta_in, beta_out)

    # 距離（AU）
    r_sun_au = float(np.linalg.norm(sc - sun))
    r_earth_au = float(np.linalg.norm(sc - earth))

    sun_aspect = angle_deg(sail_n, sun_dir)

    # 発電：太陽方向との内積（cos）で 0〜100%
    pwr = power_percent(sail_n, sun_dir)

    # 通信：地球方向が「±帆法線」から 60°以内ならOK（それ以外はNG）
    comm = comm_ok(sail_n, earth_dir)
    comm_q = comm_strength_0_100(sail_n, earth_dir)

    # 通信できたら推定が良くなる（測位アップデートの模型）
    if comm:
        st.session_state.x_hat = st.session_state.x_true.copy()

    st.session_state.log.append({
        "turn": float(turn + 1),
        "day": float(day),
        "beta_in": float(beta_in),
        "beta_out": float(beta_out),
        "pwr": float(pwr),
        "comm": float(1.0 if comm else 0.0),
        "comm_q": float(comm_q),
        "x_true_BT": float(st.session_state.x_true[0]),
        "x_true_BR": float(st.session_state.x_true[1]),
        "x_hat_BT": float(st.session_state.x_hat[0]),
        "x_hat_BR": float(st.session_state.x_hat[1]),
        "drift_BT": float(d[0]),
        "drift_BR": float(d[1]),
    })

    # 時間を進める
    st.session_state.turn = turn + 1
    st.session_state.day = float(st.session_state.day) + C.total_days / (C.n_turns - 1)

    st.session_state.history.append(st.session_state.x_hat.copy())


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="IKAROS-GO! (Full)", layout="wide")
st.title("IKAROS-GO!（本格・決定論版）")
st.caption(f"Build: {APP_BUILD} / 乱数なし（同じ操作→同じ結果）")

# Streamlitは「アプリ更新後もセッションの中身が残る」ことがあります。
# その場合、古い版の変数だけ残って新しい変数が無い → エラー、が起きがちです。
# ここで「必要な変数が全部そろっているか」をチェックして、足りなければ初期化します。
REQUIRED_KEYS = [
    "turn", "day", "k_true", "k_hat",
    "x_true", "x_hat", "history", "log",
]

def ensure_state() -> None:
    missing = [k for k in REQUIRED_KEYS if k not in st.session_state]
    if missing:
        init_state()

ensure_state()

with st.sidebar:
    st.header("ゲーム")
    st.write("**5ターン**で、目標（★）に近づけます。")
    if st.button("リセット（最初から）"):
        init_state()
        st.rerun()

    st.divider()
    st.header("操作（このターン）")
    beta_in = st.slider("β_in（deg）", -C.beta_in_deg_lim, C.beta_in_deg_lim, 0.0, 0.5)
    beta_out = st.slider("β_out（deg）", -C.beta_out_deg_lim, C.beta_out_deg_lim, 0.0, 0.5)

    turn = int(st.session_state.turn)
    can_step = turn < (C.n_turns - 1)
    if st.button("実行！（このターンを進める）", disabled=not can_step):
        step_once(beta_in, beta_out)
        st.rerun()

    st.header("本格っぽい中身")
    st.write("状態遷移（模型）")
    st.code("x_{k+1} = x_k + drift + k · C(k) · [β_in, β_out]^T", language="text")
    st.write("このターンの C(k)（km/deg）")
    st.write(get_sensitivity(int(st.session_state.turn)))

    st.write(f"k_true（本当）={st.session_state['k_true']:.2f} / k_hat（想定）={st.session_state['k_hat']:.2f}")
    st.write("※通信できたら推定が一気に良くなる（模型）")


turn = int(st.session_state.turn)
day = float(st.session_state.day)
x_true = np.array(st.session_state.x_true, dtype=float)
x_hat = np.array(st.session_state.x_hat, dtype=float)
target = np.array(C.target_bt_br_km, dtype=float)
history = np.array(st.session_state.history, dtype=float)

# 幾何
sun, earth, venus, sc = get_positions_3d(day, C.total_days)
sun_dir = unit(sun - sc)
earth_dir = unit(earth - sc)
sail_n_now = sail_normal(sc, sun, earth, beta_in, beta_out)

# 距離（AU）
r_sun_au = float(np.linalg.norm(sc - sun))
r_earth_au = float(np.linalg.norm(sc - earth))

sun_aspect_now = angle_deg(sail_n_now, sun_dir)

    # 発電：太陽方向との内積（cos）で 0〜100%
pwr_now = power_percent(sail_n_now, sun_dir)

    # 通信：地球方向が「±帆法線」から 60°以内ならOK（それ以外はNG）
comm_now = comm_ok(sail_n_now, earth_dir)
comm_rate_now = comm_strength_0_100(sail_n_now, earth_dir)

# 予測（次）
C_hat = st.session_state.k_hat * get_sensitivity(turn)
u = np.array([beta_in, beta_out], dtype=float)
pred_next = x_hat + drift_vec(turn) + C_hat @ u
pred_r = 2200.0  # ばらつき表示（固定の目安）

# メトリクス
d_now = float(np.linalg.norm(x_hat - target))
score_now = score_from_distance(d_now)

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("ターン", f"{turn+1}/{C.n_turns}")
c2.metric("経過日数", f"{day:.0f} / {C.total_days:.0f} 日")
c3.metric("距離（推定）", f"{d_now:,.0f} km")
c4.metric("スコア（目安）", f"{score_now} 点")
c5.metric("発電量（0-100%）", f"{pwr_now:.0f} %")
c6.metric("通信", "OK ✅" if comm_now else "NG ❌")
c7.metric("通信の強さ（0-100）", f"{comm_rate_now:.0f} %")


tabs = st.tabs(["B-plane（ダーツ盤）", "太陽系2D（雰囲気）", "βマップ（発電/通信）", "3D（ベクトル）"])

with tabs[0]:
    st.subheader("B-plane（ねらいの平面）")
    st.write("★に近いほど高得点。点線の円は「許容誤差」です。")

    fig = bplane_figure(
        x_hat=x_hat,
        x_true=x_true,
        target=target,
        pred_next=pred_next,
        pred_circle_r=pred_r,
        history=history,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info("この版は『乱数なし』です。ズレは“ドリフト”と“モデル差(k_true≠k_hat)”で起きます。")

    if turn == C.n_turns - 1:
        d_fin = float(np.linalg.norm(x_hat - target))
        st.divider()
        st.subheader("結果！")
        st.write(f"目標からの距離（推定）：**{d_fin:,.0f} km**")
        st.write(f"スコア：**{score_from_distance(d_fin)} 点**")

with tabs[1]:
    st.subheader("太陽・地球・金星・IKAROS（2Dの雰囲気）")

    days = np.linspace(0.0, C.total_days, 220)
    E = []
    V = []
    S = []
    for dd in days:
        sun2, earth2, venus2, sc2 = get_positions_3d(float(dd), C.total_days)
        E.append(earth2)
        V.append(venus2)
        S.append(sc2)
    E = np.array(E); V = np.array(V); S = np.array(S)

    # いまの時刻まで＝実行済み / これから先＝未実行
    idx_now = int(np.searchsorted(days, day))
    idx_now = max(1, min(idx_now, len(days)-1))

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=E[:idx_now,0], y=E[:idx_now,1], mode="lines", name="地球（実行済み）", line=dict(dash="solid")))
    fig2.add_trace(go.Scatter(x=E[idx_now:,0], y=E[idx_now:,1], mode="lines", name="地球（未実行）", line=dict(dash="dash")))
    fig2.add_trace(go.Scatter(x=V[:idx_now,0], y=V[:idx_now,1], mode="lines", name="金星（実行済み）", line=dict(dash="solid")))
    fig2.add_trace(go.Scatter(x=V[idx_now:,0], y=V[idx_now:,1], mode="lines", name="金星（未実行）", line=dict(dash="dash")))
    fig2.add_trace(go.Scatter(x=S[:idx_now,0], y=S[:idx_now,1], mode="lines", name="IKAROS（実行済み）", line=dict(dash="solid")))
    fig2.add_trace(go.Scatter(x=S[idx_now:,0], y=S[idx_now:,1], mode="lines", name="IKAROS（未実行）", line=dict(dash="dash")))

    fig2.add_trace(go.Scatter(x=[0.0], y=[0.0], mode="markers+text", name="太陽", text=["☀"], textposition="top center"))
    fig2.add_trace(go.Scatter(x=[earth[0]], y=[earth[1]], mode="markers+text", name="地球（いま）", text=["🌍"], textposition="top center"))
    fig2.add_trace(go.Scatter(x=[venus[0]], y=[venus[1]], mode="markers+text", name="金星（いま）", text=["♀"], textposition="top center"))
    fig2.add_trace(go.Scatter(x=[sc[0]], y=[sc[1]], mode="markers+text", name="IKAROS（いま）", text=["🚀"], textposition="top center"))

    fig2.update_layout(height=650, margin=dict(l=10,r=10,t=10,b=10), xaxis_title="x [AU]", yaxis_title="y [AU]")
    fig2.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig2, use_container_width=True)

    st.write("※これは“雰囲気”の絵です（本当の軌道計算ではありません）。")

with tabs[2]:
    st.subheader("βマップ（発電：cos / 通信：60°ルール）")
    st.write("いまの幾何（太陽・地球の方向）で、β_in/out を動かしたときの発電と通信を見ます。")

    xs = np.linspace(-C.beta_in_deg_lim, C.beta_in_deg_lim, 71)
    ys = np.linspace(-C.beta_out_deg_lim, C.beta_out_deg_lim, 71)

    P = np.zeros((len(ys), len(xs)), dtype=float)
    COMM = np.zeros((len(ys), len(xs)), dtype=float)
    COMM_Q = np.zeros((len(ys), len(xs)), dtype=float)

    for j, bo in enumerate(ys):
        for i, bi in enumerate(xs):
            n = sail_normal(sc, sun, earth, float(bi), float(bo))

            # 発電：太陽との内積（cos）で 0〜100%
            P[j, i] = power_percent(n, sun_dir)

            # 通信：60°以内ならOK（1） / それ以外はNG（0）
            ok = comm_ok(n, earth_dir)
            COMM[j, i] = 1.0 if ok else 0.0
            COMM_Q[j, i] = comm_strength_0_100(n, earth_dir)

    fig3 = go.Figure()
    fig3.add_trace(go.Heatmap(x=xs, y=ys, z=P, colorbar=dict(title="発電(%)")))
    fig3.add_trace(go.Scatter(x=[beta_in], y=[beta_out], mode="markers", name="いまのβ"))
    fig3.update_layout(xaxis_title="β_in (deg)", yaxis_title="β_out (deg)", height=650, margin=dict(l=10,r=10,t=10,b=10))
    fig3.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("#### 通信のβマップ（60°以内ならOK）")
    fig3a = go.Figure()
    fig3a.add_trace(go.Heatmap(x=xs, y=ys, z=COMM, colorbar=dict(title="通信OK(0/1)")))
    fig3a.add_trace(go.Scatter(x=[beta_in], y=[beta_out], mode="markers", name="いまのβ"))
    fig3a.update_layout(xaxis_title="β_in (deg)", yaxis_title="β_out (deg)", height=650, margin=dict(l=10,r=10,t=10,b=10))
    fig3a.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig3a, use_container_width=True)

    st.markdown("#### 通信のβマップ（強さ）")
    fig3b = go.Figure()
    fig3b.add_trace(go.Heatmap(x=xs, y=ys, z=COMM_Q, colorbar=dict(title="通信(0-100)")))
    fig3b.add_trace(go.Scatter(x=[beta_in], y=[beta_out], mode="markers", name="いまのβ"))
    fig3b.update_layout(xaxis_title="β_in (deg)", yaxis_title="β_out (deg)", height=650, margin=dict(l=10,r=10,t=10,b=10))
    fig3b.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig3b, use_container_width=True)

with tabs[3]:
    st.subheader("3D（ベクトル：太陽/地球/帆法線/通信コーン）")
    st.write("軌道線は描かず、いまの姿勢と方向ベクトルだけ表示します。")

    # IKAROSの見た目：画像を入れると“それっぽく”なります（任意）
    up = st.file_uploader("IKAROS画像（任意：PNG/JPG）", type=["png", "jpg", "jpeg"])
    tex_img = None
    if up is not None:
        try:
            tex_img = Image.open(io.BytesIO(up.read())).convert("RGB")  # カラー
        except Exception:
            tex_img = None

    # 画像をアップロードしなかった場合は、同梱のデフォルト画像を使う
    if tex_img is None and DEFAULT_TEX_PATH.exists():
        try:
            tex_img = Image.open(DEFAULT_TEX_PATH).convert("RGB")
        except Exception:
            tex_img = None

    sail_n = sail_n_now

    fig4 = go.Figure()

    # ---- IKAROSの帆（薄い板）を3Dで表示（カラー画像テクスチャ風） ----
    axis = unit(sail_n)
    a = np.array([0.0, 0.0, 1.0])
    b1 = np.cross(axis, a)
    if norm(b1) < 1e-6:
        a = np.array([0.0, 1.0, 0.0])
        b1 = np.cross(axis, a)
    b1 = unit(b1)
    b2 = unit(np.cross(axis, b1))

    # 帆のサイズ（見た目用）
    sail_half = 0.82

    # メッシュ分割数（大きいほど綺麗だけど重い）
    nu, nv = 36, 36
    uu = np.linspace(-1.0, 1.0, nu)
    vv = np.linspace(-1.0, 1.0, nv)

    # 画像（カラー）をメッシュに合わせてリサイズ
    if tex_img is not None:
        img2 = tex_img.resize((nu, nv))
        tex = np.array(img2, dtype=np.uint8)  # (nv, nu, 3)
    else:
        # 画像が無いときは簡易模様（カラー）
        U, V = np.meshgrid(np.linspace(-1, 1, nu), np.linspace(-1, 1, nv))
        base = 180 + 60 * (np.exp(-((U*4)**2)) + np.exp(-((V*4)**2)))
        base = np.clip(base, 0, 255).astype(np.uint8)
        tex = np.stack([base, base, base], axis=-1)

    # 頂点座標
    Xv, Yv, Zv = [], [], []
    colors = []
    for j, v in enumerate(vv):
        for i, u in enumerate(uu):
            p = sail_half * (u * b1 + v * b2)
            Xv.append(float(p[0])); Yv.append(float(p[1])); Zv.append(float(p[2]))
            r, g, b = tex[j, i, 0], tex[j, i, 1], tex[j, i, 2]
            colors.append(f"rgb({int(r)},{int(g)},{int(b)})")

    # 三角形分割（2枚/セル）
    I, J, K = [], [], []
    def vid(ii, jj): return jj * nu + ii
    for jj in range(nv - 1):
        for ii in range(nu - 1):
            v00 = vid(ii, jj)
            v10 = vid(ii + 1, jj)
            v01 = vid(ii, jj + 1)
            v11 = vid(ii + 1, jj + 1)
            I += [v00, v10]
            J += [v10, v11]
            K += [v01, v01]

    fig4.add_trace(go.Mesh3d(
        x=Xv, y=Yv, z=Zv,
        i=I, j=J, k=K,
        vertexcolor=colors,
        name="IKAROS帆",
        opacity=1.0,
        hoverinfo="skip",
    ))

    # 本体（小さい点）
    fig4.add_trace(go.Scatter3d(
        x=[0.0], y=[0.0], z=[0.0],
        mode="text",
        text=["IKAROS"],
        textposition="top center",
        name="本体",
    ))


    def add_vec(v: np.ndarray, name: str):
        vv = unit(v)
        L = 1.05  # ベクトルの表示長さ（見た目用）
        tip = L * vv

        # 線（マーカー無し）
        fig4.add_trace(go.Scatter3d(
            x=[0, tip[0]], y=[0, tip[1]], z=[0, tip[2]],
            mode="lines", name=name
        ))

        # 矢印の“先端だけ”を tip の位置に置く（原点に来ないようにする）
        head_len = 0.35
        fig4.add_trace(go.Cone(
            x=[float(tip[0])], y=[float(tip[1])], z=[float(tip[2])],
            u=[float(head_len * vv[0])], v=[float(head_len * vv[1])], w=[float(head_len * vv[2])],
            anchor="tip",
            showscale=False,
            sizemode="absolute",
            sizeref=0.22,
            name=name + "（矢印）",
            showlegend=False,
            hoverinfo="skip",
            opacity=0.9,
        ))
    add_vec(sun_dir, "太陽方向")
    add_vec(earth_dir, "地球方向")
    add_vec(sail_n, "帆面法線（アンテナ向き）")

    def cone_surface(axis: np.ndarray, half_deg: float, length: float, ntheta: int = 60, ns: int = 30):
        """
        コーン（側面）を作る：半球にならないように“側面だけ”描きます。
        - half_deg：コーンの半角
        - length：コーンの長さ
        """
        axis = unit(axis)
        a = np.array([0.0, 0.0, 1.0])
        e1 = np.cross(axis, a)
        if norm(e1) < 1e-6:
            a = np.array([0.0, 1.0, 0.0])
            e1 = np.cross(axis, a)
        e1 = unit(e1)
        e2 = unit(np.cross(axis, e1))

        th = np.linspace(0, 2 * math.pi, ntheta)
        ss = np.linspace(0.0, float(length), ns)
        p = math.radians(float(half_deg))

        X = np.zeros((ns, ntheta))
        Y = np.zeros((ns, ntheta))
        Z = np.zeros((ns, ntheta))
        for i, s in enumerate(ss):
            for j, t in enumerate(th):
                d = (math.cos(p) * axis +
                     math.sin(p) * (math.cos(t) * e1 + math.sin(t) * e2))
                d = unit(d) * s
                X[i, j], Y[i, j], Z[i, j] = d[0], d[1], d[2]
        return X, Y, Z

    X, Y, Z = cone_surface(sail_n, C.comm_cone_half_deg, 0.9, 50, 40)
    fig4.add_trace(go.Surface(x=X, y=Y, z=Z, showscale=False, opacity=0.2, name="通信コーン（表）"))

    X2, Y2, Z2 = cone_surface(-sail_n, C.comm_cone_half_deg, 0.9, 50, 40)
    fig4.add_trace(go.Surface(x=X2, y=Y2, z=Z2, showscale=False, opacity=0.2, name="通信コーン（裏）"))

    fig4.update_layout(
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="cube",
        ),
        height=700,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### かんたん説明")
    st.write(
        "- **帆面法線（アンテナ向き）**を中心に、通信コーンを置いています。\n"
        "  地球方向ベクトルがコーンの中に入ると通信しやすい（このゲームのルール）\n"
        "- **太陽方向**に近いほど、発電が増える\n"
        "- ここでは軌道の線は描かず、“向き”だけを見ます\n"
        "- 画像をアップロードすると、帆の見た目が画像っぽくなります（カラー表示）"
    )

with st.expander("ログ（確認用）"):
    if st.session_state.log:
        st.json(st.session_state.log)
    else:
        st.write("まだログはありません。")

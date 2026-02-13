# IKAROS-GO! (Darts-first / S-mode)
# Streamlit app - educational parody game inspired by IKAROS solar-sail guidance.
#
# ねらい：
# - 「膜（帆）の向きを変えると、軌道が変わる！」を体験できる
# - 難しいB-planeの理屈は“裏側”にして、ゲームとして気持ちよく遊べることを優先する
#
# 操作（IKAROSに立ち返る）
# - 開き量：調整しない（面積は固定だと思ってOK）
# - α（アルファ）：太陽にどれくらい正面？ → 押す強さ（効き）
# - β（ベータ）：どっち方向に押す？ → B-plane上の移動方向
#
# 将来の拡張（あとで入れやすいように）
# - 通信/発電の制約、βマップ、3D可視化、データ生成…などは別モードとして追加可能

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, List, Dict

import numpy as np
import streamlit as st
import plotly.graph_objects as go

APP_BUILD = "v9-darts-S-2026-02-14"


# ----------------------------
# 基本ユーティリティ
# ----------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / (n + 1e-12)


# ----------------------------
# ゲーム設定
# ----------------------------
@dataclass
class DartsConfig:
    # ターン数（固定：5ターン）
    n_turns: int = 5

    # 盤面スケール（0点になる半径）
    score_radius_km: float = 60000.0

    # 金星の大きさ（円の表示用：半径[km]）
    venus_radius_km: float = 6052.0

    # 1ターンで動く“基本量”（強さが最大のときの平均移動量）
    step_km: float = 12000.0

    # 「ゆっくりしたクセ（風）」：ゲーム中は一定のズレとして入る（見える化する）
    wind_km: float = 1200.0  # 大きいほど“癖”が強い

    # 「ばらつき」：毎ターンのランダム（予測丸の半径に対応）
    noise_sigma_km: float = 1800.0

    # 目標点（ダーツの中心）
    target_bt_br_km: Tuple[float, float] = (25000.0, -12000.0)


CFG = DartsConfig()


# ----------------------------
# 物理っぽい（でもゲーム優先）モデル
# ----------------------------
def eff_from_alpha_deg(alpha_deg: float) -> float:
    """
    α（アルファ）：太陽にどれくらい正面？
    - 0°：まっすぐ（強い）
    - 大きい：傾く（弱い）
    ゲーム用の簡単モデル：効き = cos^2(alpha)
    """
    a = math.radians(clamp(alpha_deg, 0.0, 75.0))
    return float(math.cos(a) ** 2)


def mean_delta_km(alpha_deg: float, beta_deg: float, step_km: float) -> np.ndarray:
    e = eff_from_alpha_deg(alpha_deg)
    b = math.radians(beta_deg)
    dir2 = np.array([math.cos(b), math.sin(b)], dtype=float)
    return (step_km * e) * dir2


def score_from_distance(d_km: float, R_km: float) -> int:
    """
    目標からの距離で点数（0〜100）
    - d=0 なら 100点
    - d>=R なら 0点
    """
    s = 100.0 * max(0.0, 1.0 - d_km / (R_km + 1e-12))
    return int(round(s))


# ----------------------------
# セッション状態
# ----------------------------
def init_state(seed: int = 1) -> None:
    rng = np.random.default_rng(seed)

    st.session_state.turn = 0
    st.session_state.pos = np.array([0.0, 0.0], dtype=float)  # 現在位置（BT,BR）
    st.session_state.history = [st.session_state.pos.copy()]

    # 風（ゲーム中は固定）
    ang = rng.uniform(0.0, 2 * math.pi)
    st.session_state.wind = CFG.wind_km * np.array([math.cos(ang), math.sin(ang)], dtype=float)

    # ログ
    st.session_state.log: List[Dict[str, float]] = []


# ----------------------------
# ダーツ盤（B-plane）描画
# ----------------------------
def add_filled_ring(fig: go.Figure, center: np.ndarray, r_in: float, r_out: float, name: str) -> None:
    """
    塗りつぶしリング（ドーナツ状）。
    外周と内周をつないだ多角形で描いて、fill="toself" で塗ります。
    """
    th = np.linspace(0.0, 2 * math.pi, 240)
    outer = np.c_[center[0] + r_out * np.cos(th), center[1] + r_out * np.sin(th)]
    inner = np.c_[center[0] + r_in * np.cos(th[::-1]), center[1] + r_in * np.sin(th[::-1])]
    poly = np.vstack([outer, inner, outer[:1]])
    fig.add_trace(go.Scatter(
        x=poly[:, 0], y=poly[:, 1],
        mode="lines",
        fill="toself",
        name=name,
        line=dict(width=1),
        opacity=0.18,
        showlegend=False,
    ))


def build_darts_figure(
    pos: np.ndarray,
    target: np.ndarray,
    score_R: float,
    venus_R: float,
    pred_mean: np.ndarray,
    pred_sigma: float,
    wind: np.ndarray,
    history: np.ndarray,
) -> go.Figure:
    fig = go.Figure()

    # ダーツ盤（塗りリング）
    rings = [
        (0.00, 0.15, "100点くらい"),
        (0.15, 0.30, "80点くらい"),
        (0.30, 0.50, "60点くらい"),
        (0.50, 0.75, "40点くらい"),
        (0.75, 1.00, "20点くらい"),
    ]
    for a, b, name in rings:
        add_filled_ring(fig, target, a * score_R, b * score_R, name)

    # 金星の大きさ（スケール感）
    th = np.linspace(0.0, 2 * math.pi, 240)
    fig.add_trace(go.Scatter(
        x=0.0 + venus_R * np.cos(th),
        y=0.0 + venus_R * np.sin(th),
        mode="lines",
        name="金星の大きさ（半径）",
        line=dict(width=2),
    ))

    # 目標点
    fig.add_trace(go.Scatter(
        x=[target[0]], y=[target[1]],
        mode="markers+text",
        name="目標（中心）",
        text=["★"],
        textposition="top center",
    ))

    # 履歴（点の列）
    fig.add_trace(go.Scatter(
        x=history[:, 0], y=history[:, 1],
        mode="markers",
        name="これまで",
    ))

    # 現在位置
    fig.add_trace(go.Scatter(
        x=[pos[0]], y=[pos[1]],
        mode="markers",
        name="いまの位置",
    ))

    # 予測矢印（平均）
    p2 = pos + pred_mean
    fig.add_trace(go.Scatter(
        x=[pos[0], p2[0]], y=[pos[1], p2[1]],
        mode="lines+markers",
        name="予測：このへんに行く（平均）",
    ))

    # 予測の丸（ばらつき）
    r = float(pred_sigma)
    fig.add_trace(go.Scatter(
        x=p2[0] + r * np.cos(th),
        y=p2[1] + r * np.sin(th),
        mode="lines",
        name="予測：ばらつき（目安）",
        line=dict(width=1, dash="dot"),
    ))

    # 風（クセ）表示：盤面の端に矢印
    anchor = target + np.array([-0.92 * score_R, -0.92 * score_R], dtype=float)
    wind_tip = anchor + 3.0 * unit(wind) * (0.08 * score_R)
    fig.add_trace(go.Scatter(
        x=[anchor[0], wind_tip[0]], y=[anchor[1], wind_tip[1]],
        mode="lines+markers",
        name="今日の宇宙のクセ（風）",
    ))

    # 体裁
    lim = score_R * 1.05
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


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="IKAROS-GO! (Darts)", layout="wide")

st.title("IKAROS-GO!：膜（帆）の向きで、ダーツを当てよう！")
st.caption(f"Build: {APP_BUILD}")

with st.sidebar:
    st.header("ゲーム")
    st.write("**5ターン**で終わります。最後に、目標に近いほど高得点！")

    seed = st.number_input("ランダムのタネ（同じにすると同じ風になる）", min_value=0, max_value=9999, value=1, step=1)

    if st.button("リセット（最初から）"):
        init_state(int(seed))

    st.divider()
    st.header("操作（このターン）")
    st.write("開き量は変えません（面積は固定）。**向き**だけ変えます。")

    alpha = st.slider("α：太陽にどれくらい正面？（小さいほど強い）", 0.0, 60.0, 15.0, 0.5)
    beta = st.slider("β：どっち方向に押す？（角度）", 0.0, 360.0, 0.0, 1.0)

    st.divider()
    st.header("あとで追加できる要素")
    st.write("- 通信・発電のルール（難易度調整）\n- βマップ\n- 3D表示\n- 実データ生成ツール")


# 初期化
if "turn" not in st.session_state:
    init_state(int(seed))

turn = int(st.session_state.turn)
pos = np.array(st.session_state.pos, dtype=float)
target = np.array(CFG.target_bt_br_km, dtype=float)
wind = np.array(st.session_state.wind, dtype=float)
history = np.array(st.session_state.history, dtype=float)

# 予測（平均とばらつき）
pred_mean = mean_delta_km(alpha, beta, CFG.step_km) + wind
pred_sigma = CFG.noise_sigma_km

# 現在スコア（「いまの位置」から目標まで）
d_now = float(np.linalg.norm(pos - target))
score_now = score_from_distance(d_now, CFG.score_radius_km)

# 上段メーター
c1, c2, c3, c4 = st.columns(4)
c1.metric("ターン", f"{turn+1}/{CFG.n_turns}")
c2.metric("いまの距離", f"{d_now:,.0f} km")
c3.metric("いまの点数（目安）", f"{score_now} 点")
c4.metric("効き（強さ）", f"{eff_from_alpha_deg(alpha)*100:.0f} %")


left, right = st.columns([1.3, 0.7], gap="large")

with left:
    st.subheader("ダーツ盤（B-plane）")
    fig = build_darts_figure(
        pos=pos,
        target=target,
        score_R=CFG.score_radius_km,
        venus_R=CFG.venus_radius_km,
        pred_mean=pred_mean,
        pred_sigma=pred_sigma,
        wind=wind,
        history=history,
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("このターンにやること")
    st.info("進めるボタンは **左のサイドバー** にあります。")
    st.write("1) αとβを決める → 2) 『実行！』 → 点が動く")

    eff = eff_from_alpha_deg(alpha)
    st.write(f"- 今回の**強さ**（効き）：{eff*100:.0f}%")
    st.write(f"- 予測の平均移動：BT {pred_mean[0]:+.0f} km, BR {pred_mean[1]:+.0f} km")
    st.write(f"- ばらつき（目安）：±{pred_sigma:.0f} km")

    can_step = turn < (CFG.n_turns - 1)
    btn = st.sidebar.button("実行！（このターンの操作を反映）", disabled=(not can_step))

    if btn:
        rng = np.random.default_rng(int(seed) + 1000 + turn)

        # 実際の移動（平均＋ばらつき）
        noise = rng.normal(0.0, CFG.noise_sigma_km, size=2)
        delta = pred_mean + noise
        pos2 = pos + delta

        st.session_state.pos = pos2
        st.session_state.turn = turn + 1
        st.session_state.history.append(pos2.copy())
        st.session_state.log.append({
            "turn": float(turn + 1),
            "alpha_deg": float(alpha),
            "beta_deg": float(beta),
            "eff": float(eff),
            "dBT": float(delta[0]),
            "dBR": float(delta[1]),
            "noiseBT": float(noise[0]),
            "noiseBR": float(noise[1]),
        })

        st.success("動かした！ 次のターンへ。")
        st.rerun()

    if turn == (CFG.n_turns - 1):
        d_fin = float(np.linalg.norm(pos - target))
        score_fin = score_from_distance(d_fin, CFG.score_radius_km)
        st.divider()
        st.subheader("結果！")
        st.write(f"目標からの距離：**{d_fin:,.0f} km**")
        st.write(f"スコア：**{score_fin} 点**")
        st.write("また遊ぶなら、左の『リセット』を押してね。")

    st.divider()
    st.subheader("メモ（小学生向け）")
    st.write(
        "- **β（ベータ）**は『どっち方向に押す？』\n"
        "- **α（アルファ）**は『太陽にどれくらい正面？（強さ）』\n"
        "- 点は、毎回ちょっとだけズレるよ（宇宙はむずかしい！）"
    )

    with st.expander("大人向け：このゲームの中身（超ざっくり）"):
        st.code(
            "Δ = K·cos²(α)·[cosβ, sinβ] + wind + noise\n"
            "score = 100·max(0, 1 - |pos-target|/R)",
            language="text"
        )

with st.expander("ログ（デバッグ用）"):
    if st.session_state.log:
        st.json(st.session_state.log[-5:])
    else:
        st.write("まだログはありません。")

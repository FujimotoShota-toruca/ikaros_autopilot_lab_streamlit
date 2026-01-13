\
# app.py
# IKAROS-GO (prototype) — Streamlit app
#
# ねらい:
#  - B-plane(ねらいの平面)で「どこに行くか」を見える化
#  - β_in / β_out を動かしたときの「つぎの場所（よそう）」を予測楕円で表示
#  - 太陽・地球・金星・IKAROS の 2D 図で「いまどこ？」を表示
#  - 軌道(カンタン模型)から幾何(きか)を計算し、β平面で「通信」と「でんき」を見える化
#
# 注意:
#  - これは “教育用のカンタン模型” です（本物の軌道伝播ではありません）
#  - あとから data/*.json に差し替えて “本物寄り” にできます（docs/customize.md）

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st


# -----------------------------
# 設定
# -----------------------------

@dataclass
class GameConfig:
    # 1ターン=2週間（運用っぽさ）
    n_turns: int = 14
    turn_days: int = 14

    # B-plane 目標
    target_bt_br_km: Tuple[float, float] = (0.0, 0.0)
    tolerance_km: float = 30.0

    # βスライダー
    beta_max_deg: float = 15.0
    beta_step_deg: float = 0.5

    # ノイズ（“わからなさ”）
    process_noise_km: float = 3.0
    meas_noise_km: float = 6.0
    init_est_noise_km: float = 10.0

    # でんき（太陽に対する傾き制約）
    sun_tilt_limit_deg: float = 45.0

    # 通信（アンテナ角）制約（カンタン版）
    # 0〜60° または 120〜180°ならOK（60〜120°はダメ）
    comm_ok_low_deg: float = 60.0
    comm_ok_high_deg: float = 120.0

    # “どうしても通信できない期間”の演出（カンタン）
    blackout_start_day: float = 100.0
    blackout_end_day: float = 130.0


CFG = GameConfig()


# -----------------------------
# 軌道（カンタン模型）
# -----------------------------
# 単位は AU（天文単位）っぽいスケール。図のための模型です。

EARTH_AU = 1.0
VENUS_AU = 0.723
EARTH_PERIOD_D = 365.25
VENUS_PERIOD_D = 224.7

def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)

def planet_pos(day: float, a_au: float, period_day: float, phase: float = 0.0) -> np.ndarray:
    theta = 2.0 * math.pi * (day / period_day) + phase
    return np.array([a_au * math.cos(theta), a_au * math.sin(theta)], dtype=float)

def sc_pos(day: float, total_days: float, earth_phase: float = 0.0) -> np.ndarray:
    """
    IKAROSの場所（カンタン模型）:
    - 半径は地球軌道(1AU)→金星軌道(0.723AU)へ、なめらかに近づく
    - 角度は地球の角度より少し速く回る（それっぽい）
    """
    f = smoothstep(day / total_days)
    r = EARTH_AU - (EARTH_AU - VENUS_AU) * f

    # 角速度（模型）：地球より少し速く
    omega = 2.0 * math.pi / 320.0  # rad/day (toy)
    theta = 2.0 * math.pi * (day / EARTH_PERIOD_D) + earth_phase + omega * day
    return np.array([r * math.cos(theta), r * math.sin(theta)], dtype=float)

def angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    u = u / (np.linalg.norm(u) + 1e-12)
    v = v / (np.linalg.norm(v) + 1e-12)
    c = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))

def geometry_alpha(day: float, total_days: float) -> float:
    """
    α = IKAROSから見た「太陽の方向」と「地球の方向」のなす角（0〜180°）
    これが小さいと、地球は太陽の近くに見えて “やりにくい” ことが多い。
    """
    sun = np.array([0.0, 0.0], dtype=float)
    earth = planet_pos(day, EARTH_AU, EARTH_PERIOD_D, phase=0.0)
    sc = sc_pos(day, total_days, earth_phase=0.0)

    v_sun = sun - sc
    v_earth = earth - sc
    return angle_deg(v_sun, v_earth)

def in_blackout(day: float) -> bool:
    return CFG.blackout_start_day <= day <= CFG.blackout_end_day


# -----------------------------
# β → でんき / 通信（カンタン幾何）
# -----------------------------

def beta_to_sun_tilt_deg(beta_in_deg: float, beta_out_deg: float) -> float:
    """太陽からどれだけ傾けたか（大きさ）"""
    return float(math.sqrt(beta_in_deg**2 + beta_out_deg**2))

def power_percent(sun_tilt_deg: float) -> float:
    """
    発電量（模型）:
    - 太陽に正面(0°)で 100%
    - 傾くほど cos で減る
    """
    rad = math.radians(sun_tilt_deg)
    return float(max(0.0, math.cos(rad)) * 100.0)

def earth_aspect_deg(alpha_deg: float, beta_in_deg: float, beta_out_deg: float) -> float:
    """
    アンテナ角（模型）:
    - IKAROSの “むいている向き” を β_in/β_out で決める
    - 地球がいる向きは、ローカル座標で [sinα, 0, cosα]
    - IKAROSの向きは、[sinβin, sinβout, cos(tilt)] を正規化したもの
    - 2つのなす角が “地球に向けられているか” の目安
    """
    tilt = beta_to_sun_tilt_deg(beta_in_deg, beta_out_deg)

    n = np.array([math.sin(math.radians(beta_in_deg)),
                  math.sin(math.radians(beta_out_deg)),
                  math.cos(math.radians(tilt))], dtype=float)
    n = n / (np.linalg.norm(n) + 1e-12)

    e = np.array([math.sin(math.radians(alpha_deg)), 0.0, math.cos(math.radians(alpha_deg))], dtype=float)
    e = e / (np.linalg.norm(e) + 1e-12)

    c = float(np.clip(np.dot(n, e), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))

def comm_ok(earth_aspect: float) -> bool:
    return (earth_aspect <= CFG.comm_ok_low_deg) or (earth_aspect >= CFG.comm_ok_high_deg)


# -----------------------------
# B-plane のダイナミクス（カンタン）
# -----------------------------

def base_sensitivity(turn: int) -> np.ndarray:
    """
    C_k（模型）:
    - 後半ほど “効きやすい”
    """
    gain = 0.6 + 0.06 * turn
    return gain * np.array([[1.0, 0.3],
                            [-0.2, 0.9]], dtype=float)

def predict_next_ellipse(x_hat: np.ndarray, turn: int, u: np.ndarray,
                         k_hat: float, process_noise_km: float,
                         n_samples: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """
    次の位置の “よそうのふくらみ（楕円）” をモンテカルロで作る。
    返り値: (平均, 共分散)
    """
    C = base_sensitivity(turn)  # (2,2)
    rng = np.random.default_rng(12345 + turn)

    # “セイルの効き”の不確かさ（模型）
    sigma_k = 0.15
    k = rng.normal(k_hat, sigma_k, size=n_samples)
    k = np.clip(k, 0.2, 2.0)

    w = rng.normal(0.0, process_noise_km, size=(n_samples, 2))
    du = (k[:, None] * (C @ u)[None, :])
    samples = x_hat[None, :] + du + w
    mu = samples.mean(axis=0)
    cov = np.cov(samples.T)
    return mu, cov

def ellipse_points(mu: np.ndarray, cov: np.ndarray, nsig: float = 1.0, n: int = 180) -> np.ndarray:
    """
    共分散から楕円の点列を作る（2D）。
    """
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-9)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    t = np.linspace(0, 2*np.pi, n)
    circle = np.vstack([np.cos(t), np.sin(t)])  # (2,n)
    axes = nsig * np.sqrt(vals)[:, None] * circle
    pts = (vecs @ axes).T + mu[None, :]
    return pts


# -----------------------------
# セッション状態
# -----------------------------

def init_state(seed: int = 2) -> None:
    rng = np.random.default_rng(seed)
    st.session_state.turn = 0
    st.session_state.day = 0.0
    st.session_state.rng_seed = int(seed)

    st.session_state.x_true = np.array([120.0, -80.0], dtype=float)
    st.session_state.x_hat = st.session_state.x_true + rng.normal(0.0, CFG.init_est_noise_km, size=2)

    st.session_state.k_true = float(rng.uniform(0.75, 1.25))
    st.session_state.k_hat = 1.0

    st.session_state.log: List[Dict[str, object]] = []


if "turn" not in st.session_state:
    init_state()


# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title="IKAROS-GO", layout="wide")
st.title("IKAROS-GO（試作）— IKAROSみたいに “ねらって” うごかすゲーム")

total_days = CFG.n_turns * CFG.turn_days
progress = min(1.0, st.session_state.turn / CFG.n_turns)

with st.sidebar:
    st.header("そうさ")
    st.caption("β（ベータ）を動かすと、IKAROSのむきが変わります。")

    seed = st.number_input("ランダムのタネ（seed）", min_value=0, max_value=9999,
                           value=int(st.session_state.rng_seed), step=1)
    if st.button("さいしょからやり直す"):
        init_state(int(seed))
        st.rerun()

    st.markdown("---")

    beta_in = st.slider("β_in（左右のかたむき） [deg]",
                        -CFG.beta_max_deg, CFG.beta_max_deg, 0.0, CFG.beta_step_deg)
    beta_out = st.slider("β_out（上下のかたむき） [deg]",
                         -CFG.beta_max_deg, CFG.beta_max_deg, 0.0, CFG.beta_step_deg)

    st.markdown("---")
    n_samples = st.slider("よそうの点の数（多いほどなめらか）", 200, 1200, 500, 100)

    step = st.button("つぎの2週間へ（すすめる）", type="primary",
                     disabled=(st.session_state.turn >= CFG.n_turns))

# 現在の幾何（α）
alpha = geometry_alpha(st.session_state.day, total_days)
tilt = beta_to_sun_tilt_deg(beta_in, beta_out)
pwr = power_percent(tilt)
ea = earth_aspect_deg(alpha, beta_in, beta_out)

# 通信できた？（このβで）
comm_success = (not in_blackout(st.session_state.day)) and (tilt <= CFG.sun_tilt_limit_deg) and comm_ok(ea)

# 進みぐあい
st.caption("進みぐあい（0% → 100%）")
st.progress(progress)

# ステータス
c1, c2, c3, c4 = st.columns(4)
c1.metric("いまのターン", f"{st.session_state.turn + 1}/{CFG.n_turns}")
c2.metric("地球と太陽の角度 α", f"{alpha:.1f}°", help="IKAROSから見た地球と太陽のはなれぐあい")
c3.metric("発電（模型）", f"{pwr:.0f}%", help="太陽に正面ほど大きい")
c4.metric("通信", "できた ✅" if comm_success else "できない ❌",
          help="アンテナ角と発電の条件をみたすと通信できる")

if in_blackout(st.session_state.day):
    st.warning("いまはブラックアウト中：なにをしても通信できません（演出）")

# すすめる（ターン更新）
if step:
    rng = np.random.default_rng(int(st.session_state.rng_seed) + 1000 + int(st.session_state.turn))
    u = np.array([beta_in, beta_out], dtype=float)

    C_true = float(st.session_state.k_true) * base_sensitivity(st.session_state.turn)
    w = rng.normal(0.0, CFG.process_noise_km, size=2)
    st.session_state.x_true = st.session_state.x_true + C_true @ u + w

    if comm_success:
        meas = st.session_state.x_true + rng.normal(0.0, CFG.meas_noise_km, size=2)
        st.session_state.x_hat = 0.6 * st.session_state.x_hat + 0.4 * meas
        st.session_state.k_hat += 0.15 * (float(st.session_state.k_true) - float(st.session_state.k_hat))

    st.session_state.log.append({
        "turn": int(st.session_state.turn),
        "day": float(st.session_state.day),
        "beta_in": float(beta_in),
        "beta_out": float(beta_out),
        "BT_hat": float(st.session_state.x_hat[0]),
        "BR_hat": float(st.session_state.x_hat[1]),
        "alpha": float(alpha),
        "tilt": float(tilt),
        "power_%": float(pwr),
        "earth_aspect": float(ea),
        "comm": bool(comm_success),
        "blackout": bool(in_blackout(st.session_state.day)),
        "k_hat": float(st.session_state.k_hat),
    })

    st.session_state.turn += 1
    st.session_state.day += float(CFG.turn_days)

    if st.session_state.turn >= CFG.n_turns:
        st.success("ミッションおわり！（試作）")

    st.rerun()


# -----------------------------
# 表示タブ
# -----------------------------

tab1, tab2, tab3 = st.tabs(["B-plane（ねらい）", "太陽系の図（いまどこ？）", "βマップ（通信とでんき）"])

# --- Tab 1: B-plane ---
with tab1:
    st.subheader("B-plane（ねらいの平面）")
    st.caption("点＝いまのよそう場所。楕円＝つぎに行きそうな “ばらつき（よそう）”")

    x_hat = np.array(st.session_state.x_hat, dtype=float)
    u = np.array([beta_in, beta_out], dtype=float)

    if st.session_state.turn < CFG.n_turns:
        mu, cov = predict_next_ellipse(
            x_hat=x_hat,
            turn=int(st.session_state.turn),
            u=u,
            k_hat=float(st.session_state.k_hat),
            process_noise_km=CFG.process_noise_km,
            n_samples=int(n_samples),
        )
        e1 = ellipse_points(mu, cov, nsig=1.0)
        e2 = ellipse_points(mu, cov, nsig=2.0)
    else:
        mu, e1, e2 = x_hat, None, None

    target = np.array(CFG.target_bt_br_km, dtype=float)
    tol = float(CFG.tolerance_km)

    fig = go.Figure()

    th = np.linspace(0, 2*np.pi, 240)
    fig.add_trace(go.Scatter(
        x=target[0] + tol*np.cos(th),
        y=target[1] + tol*np.sin(th),
        mode="lines",
        name="ゴール（ゆるい範囲）",
    ))

    if st.session_state.log:
        xs = [d["BT_hat"] for d in st.session_state.log]
        ys = [d["BR_hat"] for d in st.session_state.log]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name="これまで"))

    fig.add_trace(go.Scatter(
        x=[x_hat[0]], y=[x_hat[1]],
        mode="markers",
        name="いま（よそう）"
    ))

    fig.add_trace(go.Scatter(
        x=[float(mu[0])], y=[float(mu[1])],
        mode="markers",
        name="つぎ（まんなか）"
    ))

    if e2 is not None:
        fig.add_trace(go.Scatter(
            x=e2[:, 0], y=e2[:, 1],
            mode="lines",
            name="つぎのよそう（2σ）",
            fill="toself",
            opacity=0.15,
        ))
    if e1 is not None:
        fig.add_trace(go.Scatter(
            x=e1[:, 0], y=e1[:, 1],
            mode="lines",
            name="つぎのよそう（1σ）",
            fill="toself",
            opacity=0.25,
        ))

    fig.add_trace(go.Scatter(
        x=[x_hat[0], float(mu[0])],
        y=[x_hat[1], float(mu[1])],
        mode="lines",
        name="うごく向き（平均）",
    ))

    fig.update_layout(
        xaxis_title="B_T (km)",
        yaxis_title="B_R (km)",
        height=600,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=True,
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("ログ（これまで）")
    st.dataframe(st.session_state.log, use_container_width=True)


# --- Tab 2: Orbit 2D diagram ---
with tab2:
    st.subheader("太陽・地球・金星・IKAROS（2D図）")
    st.caption("これは “それっぽい模型” の図です。ほんとうの軌道とは少しちがいます。")

    day = float(st.session_state.day)
    sun = np.array([0.0, 0.0], dtype=float)
    earth = planet_pos(day, EARTH_AU, EARTH_PERIOD_D, phase=0.0)
    venus = planet_pos(day, VENUS_AU, VENUS_PERIOD_D, phase=0.6)  # 見やすさで位相をずらす
    sc = sc_pos(day, total_days, earth_phase=0.0)

    days = np.linspace(0, total_days, 240)
    sc_path = np.vstack([sc_pos(float(d), total_days) for d in days])

    th = np.linspace(0, 2*np.pi, 360)
    earth_orbit = np.vstack([EARTH_AU*np.cos(th), EARTH_AU*np.sin(th)]).T
    venus_orbit = np.vstack([VENUS_AU*np.cos(th), VENUS_AU*np.sin(th)]).T

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(x=earth_orbit[:,0], y=earth_orbit[:,1], mode="lines", name="地球の道（円）"))
    fig2.add_trace(go.Scatter(x=venus_orbit[:,0], y=venus_orbit[:,1], mode="lines", name="金星の道（円）"))
    fig2.add_trace(go.Scatter(x=sc_path[:,0], y=sc_path[:,1], mode="lines", name="IKAROSの道（模型）"))

    fig2.add_trace(go.Scatter(x=[sun[0]], y=[sun[1]], mode="markers", name="太陽"))
    fig2.add_trace(go.Scatter(x=[earth[0]], y=[earth[1]], mode="markers", name="地球"))
    fig2.add_trace(go.Scatter(x=[venus[0]], y=[venus[1]], mode="markers", name="金星"))
    fig2.add_trace(go.Scatter(x=[sc[0]], y=[sc[1]], mode="markers", name="IKAROS（いま）"))

    fig2.update_layout(
        xaxis_title="x (AU)",
        yaxis_title="y (AU)",
        height=620,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=True,
    )
    fig2.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### いまの説明")
    st.write(
        f"- きょうは **{day:.0f}日目**（模型）\n"
        f"- 進みぐあいは **{progress*100:.0f}%**\n"
        f"- 地球と太陽の角度 α は **{alpha:.1f}°**（小さいとむずかしいことが多い）"
    )


# --- Tab 3: beta plane map ---
with tab3:
    st.subheader("βマップ（通信とでんき）")
    st.caption("β_in と β_out をどこにすると “通信できる / でんきが多い” かを見える化します。")

    grid_step = 1.0  # deg（軽さ優先）
    xs = np.arange(-CFG.beta_max_deg, CFG.beta_max_deg + 1e-9, grid_step)
    ys = np.arange(-CFG.beta_max_deg, CFG.beta_max_deg + 1e-9, grid_step)

    P = np.zeros((len(ys), len(xs)), dtype=float)
    COMM = np.zeros((len(ys), len(xs)), dtype=float)

    if in_blackout(st.session_state.day):
        for j, bo in enumerate(ys):
            for i, bi in enumerate(xs):
                tilt_ = beta_to_sun_tilt_deg(float(bi), float(bo))
                P[j, i] = power_percent(tilt_)
                COMM[j, i] = 0.0
    else:
        for j, bo in enumerate(ys):
            for i, bi in enumerate(xs):
                tilt_ = beta_to_sun_tilt_deg(float(bi), float(bo))
                P[j, i] = power_percent(tilt_)
                ea_ = earth_aspect_deg(alpha, float(bi), float(bo))
                ok = (tilt_ <= CFG.sun_tilt_limit_deg) and comm_ok(ea_)
                COMM[j, i] = 1.0 if ok else 0.0

    fig3 = go.Figure()

    fig3.add_trace(go.Heatmap(
        x=xs, y=ys, z=P,
        colorbar=dict(title="でんき(%)"),
        name="でんき"
    ))

    fig3.add_trace(go.Contour(
        x=xs, y=ys, z=COMM,
        showscale=False,
        contours=dict(start=0.5, end=0.5, size=1),
        name="通信OKの線",
        line=dict(width=3),
        opacity=0.9
    ))

    th = np.linspace(0, 2*np.pi, 240)
    r = CFG.sun_tilt_limit_deg
    fig3.add_trace(go.Scatter(
        x=r*np.cos(th),
        y=r*np.sin(th),
        mode="lines",
        name=f"でんき制限 {CFG.sun_tilt_limit_deg:.0f}°"
    ))

    fig3.add_trace(go.Scatter(
        x=[beta_in], y=[beta_out],
        mode="markers",
        name="いまのβ"
    ))

    fig3.update_layout(
        xaxis_title="β_in (deg)",
        yaxis_title="β_out (deg)",
        height=650,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=True,
    )
    fig3.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### いまのβの結果（カンタン説明）")
    st.write(
        f"- 太陽からのかたむき: **{tilt:.1f}°**（小さいほどでんき↑）\n"
        f"- でんき（模型）: **{pwr:.0f}%**\n"
        f"- 地球に向けた角度（アンテナ角）: **{ea:.1f}°**\n"
        f"- 通信: **{'できる' if comm_success else 'できない'}**"
        + ("（ブラックアウト中）" if in_blackout(st.session_state.day) else "")
    )

    st.info(
        "ポイント：\n"
        "- 背景は “でんき” です（100%が強い）。\n"
        "- 太い線の内側/外側の一部が “通信できる” ところです。\n"
        "- まるい線の外に出ると、でんきの制限をこえやすいです。"
    )

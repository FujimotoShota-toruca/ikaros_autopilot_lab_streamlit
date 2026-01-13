# app.py
# IKAROS-GO (prototype) — Streamlit app (v3: JSONデータ差し替え対応)
#
# ✅ data/orbit_schedule.json があれば、2D軌道図と幾何(太陽/地球方向)をそのデータで計算
# ✅ data/sensitivity_schedule.json があれば、β→B-plane の効き(C_k)をそのデータで置き換え
# ✅ どちらも無い場合は、カンタン模型（トイ）で動く

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = Path(__file__).parent / "data"


# -----------------------------
# 設定
# -----------------------------
@dataclass
class GameConfig:
    n_turns: int = 14
    turn_days: int = 14

    target_bt_br_km: Tuple[float, float] = (0.0, 0.0)
    tolerance_km: float = 30.0

    beta_max_deg: float = 15.0
    beta_step_deg: float = 0.5

    process_noise_km: float = 3.0
    meas_noise_km: float = 6.0
    init_est_noise_km: float = 10.0

    sun_tilt_limit_deg: float = 45.0

    comm_ok_low_deg: float = 60.0
    comm_ok_high_deg: float = 120.0

    blackout_start_day: float = 100.0
    blackout_end_day: float = 130.0


CFG = GameConfig()


# -----------------------------
# ベクトル便利関数
# -----------------------------
def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))

def unit(v: np.ndarray) -> np.ndarray:
    n = norm(v)
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return v / n

def angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    u = unit(u)
    v = unit(v)
    c = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))

def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


# -----------------------------
# JSON読み込み
# -----------------------------
def load_json(path: Path) -> Optional[object]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

ORBIT_DATA = load_json(DATA_DIR / "orbit_schedule.json")
SENS_DATA = load_json(DATA_DIR / "sensitivity_schedule.json")

def orbit_available() -> bool:
    return isinstance(ORBIT_DATA, list) and len(ORBIT_DATA) >= 2

def sens_available() -> bool:
    return isinstance(SENS_DATA, list) and len(SENS_DATA) >= 1

def _get_entry_by_day(day: float) -> Optional[Dict[str, object]]:
    if not orbit_available():
        return None
    best = min(ORBIT_DATA, key=lambda d: abs(float(d.get("day", 0.0)) - day))
    return best if isinstance(best, dict) else None

def _to_xy(entry: Dict[str, object], key: str) -> Optional[np.ndarray]:
    v = entry.get(key, None)
    if not (isinstance(v, list) and len(v) == 2):
        return None
    return np.array([float(v[0]), float(v[1])], dtype=float)


# -----------------------------
# 2D軌道（データ or トイ）
# -----------------------------
def get_positions_2d(day: float, total_days: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if orbit_available():
        e = _get_entry_by_day(day)
        if e is not None:
            sun = _to_xy(e, "sun")
            earth = _to_xy(e, "earth")
            venus = _to_xy(e, "venus")
            ikaros = _to_xy(e, "ikaros")
            if sun is not None and earth is not None and venus is not None and ikaros is not None:
                return sun, earth, venus, ikaros

    # ---- fallback toy ----
    EARTH_AU = 1.0
    VENUS_AU = 0.723
    EARTH_PERIOD_D = 365.25
    VENUS_PERIOD_D = 224.7

    def planet_pos(day_: float, a_au: float, period_day: float, phase: float = 0.0) -> np.ndarray:
        th = 2.0 * math.pi * (day_ / period_day) + phase
        return np.array([a_au * math.cos(th), a_au * math.sin(th)], dtype=float)

    def sc_pos(day_: float) -> np.ndarray:
        f = smoothstep(day_ / total_days)
        r = EARTH_AU - (EARTH_AU - VENUS_AU) * f
        omega = 2.0 * math.pi / 320.0
        th = 2.0 * math.pi * (day_ / EARTH_PERIOD_D) + omega * day_
        return np.array([r * math.cos(th), r * math.sin(th)], dtype=float)

    sun = np.array([0.0, 0.0], dtype=float)
    earth = planet_pos(day, EARTH_AU, EARTH_PERIOD_D, phase=0.0)
    venus = planet_pos(day, VENUS_AU, VENUS_PERIOD_D, phase=0.6)
    ikaros = sc_pos(day)
    return sun, earth, venus, ikaros


# -----------------------------
# 感度行列 C_k（データ or トイ）
# -----------------------------
def get_sensitivity(turn: int) -> np.ndarray:
    if sens_available():
        for d in SENS_DATA:
            if isinstance(d, dict) and int(d.get("turn", -1)) == int(turn):
                C = d.get("C", None)
                if isinstance(C, list) and len(C) == 2:
                    try:
                        M = np.array(C, dtype=float)
                        if M.shape == (2, 2):
                            return M
                    except Exception:
                        pass
    gain = 0.6 + 0.06 * turn
    return gain * np.array([[1.0, 0.3], [-0.2, 0.9]], dtype=float)


# -----------------------------
# 通信 / でんき（幾何：データ対応）
# -----------------------------
def in_blackout(day: float) -> bool:
    return CFG.blackout_start_day <= day <= CFG.blackout_end_day

def beta_to_sun_tilt_deg(bi: float, bo: float) -> float:
    return float(math.sqrt(bi * bi + bo * bo))

def power_percent(tilt_deg: float) -> float:
    return float(max(0.0, math.cos(math.radians(tilt_deg))) * 100.0)

def comm_ok(earth_aspect: float) -> bool:
    return (earth_aspect <= CFG.comm_ok_low_deg) or (earth_aspect >= CFG.comm_ok_high_deg)

def make_local_frame(sc3: np.ndarray, sun3: np.ndarray, earth3: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = unit(sun3 - sc3)
    e = unit(earth3 - sc3)
    x_raw = e - np.dot(e, z) * z
    if norm(x_raw) < 1e-6:
        tmp = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(np.dot(tmp, z)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0], dtype=float)
        x_raw = tmp - np.dot(tmp, z) * z
    x = unit(x_raw)
    y = unit(np.cross(z, x))
    return x, y, z

def earth_aspect_from_beta(sc2: np.ndarray, sun2: np.ndarray, earth2: np.ndarray, bi: float, bo: float) -> float:
    sc3 = np.array([sc2[0], sc2[1], 0.0], dtype=float)
    sun3 = np.array([sun2[0], sun2[1], 0.0], dtype=float)
    earth3 = np.array([earth2[0], earth2[1], 0.0], dtype=float)

    x, y, z = make_local_frame(sc3, sun3, earth3)

    tilt = beta_to_sun_tilt_deg(bi, bo)
    n_local = unit(np.array([math.sin(math.radians(bi)),
                             math.sin(math.radians(bo)),
                             math.cos(math.radians(tilt))], dtype=float))

    e_dir = unit(earth3 - sc3)
    e_local = unit(np.array([np.dot(e_dir, x), np.dot(e_dir, y), np.dot(e_dir, z)], dtype=float))

    return angle_deg(n_local, e_local)

def alpha_deg(sc2: np.ndarray, sun2: np.ndarray, earth2: np.ndarray) -> float:
    v_sun = np.array([sun2[0]-sc2[0], sun2[1]-sc2[1], 0.0], dtype=float)
    v_earth = np.array([earth2[0]-sc2[0], earth2[1]-sc2[1], 0.0], dtype=float)
    return angle_deg(v_sun, v_earth)


# -----------------------------
# 予測楕円（モンテカルロ）
# -----------------------------
def predict_next_ellipse(x_hat: np.ndarray, C: np.ndarray, u: np.ndarray,
                         k_hat: float, process_noise_km: float, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(12345)
    sigma_k = 0.15
    k = np.clip(rng.normal(k_hat, sigma_k, size=n_samples), 0.2, 2.0)
    w = rng.normal(0.0, process_noise_km, size=(n_samples, 2))
    du = (k[:, None] * (C @ u)[None, :])
    samples = x_hat[None, :] + du + w
    mu = samples.mean(axis=0)
    cov = np.cov(samples.T)
    return mu, cov

def ellipse_points(mu: np.ndarray, cov: np.ndarray, nsig: float, n: int = 180) -> np.ndarray:
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-9)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    t = np.linspace(0, 2*np.pi, n)
    circle = np.vstack([np.cos(t), np.sin(t)])
    axes = nsig * np.sqrt(vals)[:, None] * circle
    pts = (vecs @ axes).T + mu[None, :]
    return pts


# -----------------------------
# セッション
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
st.title("IKAROS-GO（試作 v3）— “本物のデータ” に差し替えられる版")

total_days = CFG.n_turns * CFG.turn_days
progress = min(1.0, st.session_state.turn / CFG.n_turns)

sun2, earth2, venus2, sc2 = get_positions_2d(st.session_state.day, total_days)
alpha_now = alpha_deg(sc2, sun2, earth2)

with st.sidebar:
    st.header("そうさ")
    st.caption("β（ベータ）を動かして、むきを変えよう。")

    st.caption("データ読み込み状況")
    st.write(f"- orbit_schedule.json: {'ある ✅' if orbit_available() else 'ない（模型）'}")
    st.write(f"- sensitivity_schedule.json: {'ある ✅' if sens_available() else 'ない（模型）'}")

    seed = st.number_input("ランダムのタネ（seed）", 0, 9999, int(st.session_state.rng_seed), 1)
    if st.button("さいしょからやり直す"):
        init_state(int(seed))
        st.rerun()

    st.markdown("---")
    beta_in = st.slider("β_in（左右）[deg]", -CFG.beta_max_deg, CFG.beta_max_deg, 0.0, CFG.beta_step_deg)
    beta_out = st.slider("β_out（上下）[deg]", -CFG.beta_max_deg, CFG.beta_max_deg, 0.0, CFG.beta_step_deg)

    st.markdown("---")
    n_samples = st.slider("よそうの点の数", 200, 2000, 800, 100)

    step = st.button("つぎの2週間へ（すすめる）", type="primary", disabled=(st.session_state.turn >= CFG.n_turns))

tilt = beta_to_sun_tilt_deg(beta_in, beta_out)
pwr = power_percent(tilt)
earth_aspect = earth_aspect_from_beta(sc2, sun2, earth2, beta_in, beta_out)

comm_success = (not in_blackout(st.session_state.day)) and (tilt <= CFG.sun_tilt_limit_deg) and comm_ok(earth_aspect)

st.caption("進みぐあい（0% → 100%）")
st.progress(progress)

c1, c2, c3, c4 = st.columns(4)
c1.metric("いまのターン", f"{st.session_state.turn + 1}/{CFG.n_turns}")
c2.metric("地球と太陽の角度 α", f"{alpha_now:.1f}°")
c3.metric("でんき（模型）", f"{pwr:.0f}%")
c4.metric("通信", "できた ✅" if comm_success else "できない ❌")

if in_blackout(st.session_state.day):
    st.warning("ブラックアウト中：なにをしても通信できません（演出）")

if step:
    rng = np.random.default_rng(int(st.session_state.rng_seed) + 1000 + int(st.session_state.turn))
    u = np.array([beta_in, beta_out], dtype=float)

    C = get_sensitivity(int(st.session_state.turn))
    C_true = float(st.session_state.k_true) * C
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
        "alpha_deg": float(alpha_now),
        "tilt_deg": float(tilt),
        "power_%": float(pwr),
        "earth_aspect_deg": float(earth_aspect),
        "comm": bool(comm_success),
        "k_hat": float(st.session_state.k_hat),
    })

    st.session_state.turn += 1
    st.session_state.day += float(CFG.turn_days)
    if st.session_state.turn >= CFG.n_turns:
        st.success("ミッションおわり！")
    st.rerun()


tab1, tab2, tab3 = st.tabs(["B-plane（ねらい）", "太陽系の図（いまどこ？）", "βマップ（通信とでんき）"])

with tab1:
    st.subheader("B-plane（ねらい）")
    st.caption("楕円＝つぎに行きそうな “ばらつき” です。")

    x_hat = np.array(st.session_state.x_hat, dtype=float)
    u = np.array([beta_in, beta_out], dtype=float)

    if st.session_state.turn < CFG.n_turns:
        C = get_sensitivity(int(st.session_state.turn))
        mu, cov = predict_next_ellipse(x_hat, C, u, float(st.session_state.k_hat), CFG.process_noise_km, int(n_samples))
        e1 = ellipse_points(mu, cov, 1.0)
        e2 = ellipse_points(mu, cov, 2.0)
    else:
        mu, e1, e2 = x_hat, None, None

    target = np.array(CFG.target_bt_br_km, dtype=float)
    tol = float(CFG.tolerance_km)

    fig = go.Figure()

    th = np.linspace(0, 2*np.pi, 240)
    fig.add_trace(go.Scatter(x=target[0] + tol*np.cos(th), y=target[1] + tol*np.sin(th),
                             mode="lines", name="ゴール（ゆるい範囲）"))

    if st.session_state.log:
        xs = [d["BT_hat"] for d in st.session_state.log]
        ys = [d["BR_hat"] for d in st.session_state.log]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name="これまで"))

    fig.add_trace(go.Scatter(x=[x_hat[0]], y=[x_hat[1]], mode="markers", name="いま（よそう）"))
    fig.add_trace(go.Scatter(x=[float(mu[0])], y=[float(mu[1])], mode="markers", name="つぎ（まんなか）"))

    if e2 is not None:
        fig.add_trace(go.Scatter(x=e2[:, 0], y=e2[:, 1], mode="lines", name="つぎ（2σ）", fill="toself", opacity=0.15))
    if e1 is not None:
        fig.add_trace(go.Scatter(x=e1[:, 0], y=e1[:, 1], mode="lines", name="つぎ（1σ）", fill="toself", opacity=0.25))

    fig.add_trace(go.Scatter(x=[x_hat[0], float(mu[0])], y=[x_hat[1], float(mu[1])],
                             mode="lines", name="うごく向き（平均）"))

    fig.update_layout(xaxis_title="B_T (km)", yaxis_title="B_R (km)", height=600,
                      margin=dict(l=10, r=10, t=10, b=10), showlegend=True)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("ログ")
    st.dataframe(st.session_state.log, use_container_width=True)

with tab2:
    st.subheader("太陽・地球・金星・IKAROS（2D図）")
    st.caption("データがあると、そのデータで動きます。")

    day = float(st.session_state.day)
    sun, earth, venus, sc = get_positions_2d(day, total_days)

    days = np.linspace(0, total_days, 260)
    sc_path, earth_path, venus_path = [], [], []
    for d in days:
        s, e, v, k = get_positions_2d(float(d), total_days)
        sc_path.append(k); earth_path.append(e); venus_path.append(v)
    sc_path = np.vstack(sc_path); earth_path = np.vstack(earth_path); venus_path = np.vstack(venus_path)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=earth_path[:,0], y=earth_path[:,1], mode="lines", name="地球の道"))
    fig2.add_trace(go.Scatter(x=venus_path[:,0], y=venus_path[:,1], mode="lines", name="金星の道"))
    fig2.add_trace(go.Scatter(x=sc_path[:,0], y=sc_path[:,1], mode="lines", name="IKAROSの道"))

    fig2.add_trace(go.Scatter(x=[sun[0]], y=[sun[1]], mode="markers", name="太陽"))
    fig2.add_trace(go.Scatter(x=[earth[0]], y=[earth[1]], mode="markers", name="地球"))
    fig2.add_trace(go.Scatter(x=[venus[0]], y=[venus[1]], mode="markers", name="金星"))
    fig2.add_trace(go.Scatter(x=[sc[0]], y=[sc[1]], mode="markers", name="IKAROS（いま）"))

    fig2.update_layout(xaxis_title="x", yaxis_title="y", height=640,
                       margin=dict(l=10, r=10, t=10, b=10), showlegend=True)
    fig2.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig2, use_container_width=True)

    st.write(f"- きょうは **{day:.0f}日目**  /  進みぐあい **{progress*100:.0f}%**  /  α **{alpha_now:.1f}°**")

with tab3:
    st.subheader("βマップ（通信とでんき）")
    st.caption("背景＝でんき、太い線＝通信できる場所です。")

    grid_step = 1.0
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
                ea_ = earth_aspect_from_beta(sc2, sun2, earth2, float(bi), float(bo))
                ok = (tilt_ <= CFG.sun_tilt_limit_deg) and comm_ok(ea_)
                COMM[j, i] = 1.0 if ok else 0.0

    fig3 = go.Figure()
    fig3.add_trace(go.Heatmap(x=xs, y=ys, z=P, colorbar=dict(title="でんき(%)"), name="でんき"))
    fig3.add_trace(go.Contour(x=xs, y=ys, z=COMM, showscale=False,
                              contours=dict(start=0.5, end=0.5, size=1),
                              name="通信OKの線", line=dict(width=3), opacity=0.9))

    th = np.linspace(0, 2*np.pi, 240)
    r = CFG.sun_tilt_limit_deg
    fig3.add_trace(go.Scatter(x=r*np.cos(th), y=r*np.sin(th), mode="lines", name=f"でんき制限 {r:.0f}°"))
    fig3.add_trace(go.Scatter(x=[beta_in], y=[beta_out], mode="markers", name="いまのβ"))

    fig3.update_layout(xaxis_title="β_in (deg)", yaxis_title="β_out (deg)", height=650,
                       margin=dict(l=10, r=10, t=10, b=10), showlegend=True)
    fig3.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig3, use_container_width=True)

    st.write(
        f"- 太陽からのかたむき: **{tilt:.1f}°**\n"
        f"- でんき: **{pwr:.0f}%**\n"
        f"- 地球に向けた角度（アンテナ角）: **{earth_aspect:.1f}°**\n"
        f"- 通信: **{'できる' if comm_success else 'できない'}**"
    )

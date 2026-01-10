# IKAROS β-GO! (Streamlit)
# ------------------------------------------------------------
# HTV GO!!-like simplicity:
# - Orbits advance automatically (planets are analytic circles).
# - Player controls ONE parameter: β (signed cone angle in 2D).
# - Mission: Guide from Earth orbit to Venus orbit and get close.
#
# Modes:
# 1) リアルタイム制御：βをその場で変えて誘導（自動進行/一時停止あり）
# 2) 事前角度指定：βを3区間だけ先に決めて一気に実行
#
# Extra:
# - 時間無制限（練習）
#
# Safety:
# - No exec/eval. Single-file app deployable to Streamlit Community Cloud.
#
# NOTE: Educational toy model (not a high-fidelity mission design tool).
# ------------------------------------------------------------

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


# -----------------------------
# Math helpers
# -----------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def rad(deg: float) -> float:
    return deg * math.pi / 180.0

def deg(rad_: float) -> float:
    return rad_ * 180.0 / math.pi

def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))

def unit(v: np.ndarray) -> np.ndarray:
    n = norm(v)
    if n <= 1e-12:
        return np.array([1.0, 0.0], dtype=float)
    return v / n

def rot90(v: np.ndarray) -> np.ndarray:
    # CCW 90 degrees
    return np.array([-v[1], v[0]], dtype=float)


# -----------------------------
# Solar system toy constants (AU, year)
# -----------------------------
MU = 4.0 * math.pi * math.pi  # AU^3 / yr^2 => Earth period = 1 yr at r=1
R_E = 1.0
R_V = 0.723  # Venus semi-major axis (AU)

def mean_motion(r: float) -> float:
    return math.sqrt(MU / (r**3))

N_E = mean_motion(R_E)
N_V = mean_motion(R_V)

# Game tuning
DT_DAYS_DEFAULT = 0.5
DT = DT_DAYS_DEFAULT / 365.25  # yr

# "Solar sail" strength (toy). Larger -> faster transfers (more game-like).
A0_DEFAULT = 0.030  # AU/yr^2 (toy)

# Win condition
VENUS_HIT_RADIUS = 0.035  # AU (~5.2 million km) toy threshold


# -----------------------------
# Sail model (single control β)
# -----------------------------
def sail_accel(r_sc: np.ndarray, beta_deg: float, a0: float) -> np.ndarray:
    """Idealized planar sail.
    - r_sc is heliocentric position.
    - β is signed cone angle (deg) rotating sail normal within the orbital plane.
        β>0 accelerates along prograde tangential direction (outward transfer),
        β<0 accelerates along retrograde tangential direction (inward transfer).
    - magnitude ∝ cos^2(|β|)
    """
    beta = rad(beta_deg)
    r_hat = unit(r_sc)         # sun->spacecraft direction (radial outward)
    t_hat = rot90(r_hat)       # prograde tangential (CCW) direction

    c = math.cos(beta)
    s = math.sin(beta)

    # Sail normal in plane (rotated from radial toward tangential)
    n_hat = c * r_hat + s * t_hat

    # Ideal sail acceleration magnitude scales with cos^2(beta)
    mag = a0 * (c * c)

    return mag * n_hat


# -----------------------------
# Dynamics integration (RK4)
# -----------------------------
def grav_accel(r: np.ndarray) -> np.ndarray:
    d = norm(r)
    return -MU * r / (d**3 + 1e-12)

def f(state: np.ndarray, beta_deg: float, a0: float) -> np.ndarray:
    # state = [x,y,vx,vy]
    r = state[:2]
    v = state[2:]
    a = grav_accel(r) + sail_accel(r, beta_deg, a0)
    return np.array([v[0], v[1], a[0], a[1]], dtype=float)

def rk4_step(state: np.ndarray, beta_deg: float, a0: float, dt: float) -> np.ndarray:
    k1 = f(state, beta_deg, a0)
    k2 = f(state + 0.5 * dt * k1, beta_deg, a0)
    k3 = f(state + 0.5 * dt * k2, beta_deg, a0)
    k4 = f(state + dt * k3, beta_deg, a0)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# -----------------------------
# Planets (analytic circles)
# -----------------------------
def planet_pos(r: float, n: float, t_yr: float, phase_rad: float) -> np.ndarray:
    ang = n * t_yr + phase_rad
    return np.array([r * math.cos(ang), r * math.sin(ang)], dtype=float)


# -----------------------------
# Game state
# -----------------------------
@dataclass
class GameConfig:
    dt_days: float
    a0: float
    venus_phase_deg: float
    time_limit_days: float
    time_unlimited: bool

@dataclass
class GameState:
    t_yr: float
    sc: np.ndarray  # [x,y,vx,vy]
    beta_deg: float
    auto: bool
    done: bool
    message: str
    best_dist: float

def init_game(cfg: GameConfig) -> GameState:
    # Spacecraft starts at Earth's position with Earth's circular velocity (prograde CCW)
    r0 = planet_pos(R_E, N_E, 0.0, 0.0)
    v0 = np.array([-R_E * N_E * math.sin(0.0), R_E * N_E * math.cos(0.0)], dtype=float)
    sc = np.array([r0[0], r0[1], v0[0], v0[1]], dtype=float)

    return GameState(
        t_yr=0.0,
        sc=sc,
        beta_deg=-35.0,   # a good starting hint for Earth->Venus
        auto=True,
        done=False,
        message="",
        best_dist=1e9,
    )


# -----------------------------
# UI text helpers
# -----------------------------
def beta_label(beta_deg: float) -> str:
    if beta_deg < -1e-6:
        return "減速（内側へ）"
    if beta_deg > 1e-6:
        return "加速（外側へ）"
    return "まっすぐ（待ち）"

def traffic_light(dist_trend: float, dist: float) -> Tuple[str, str]:
    """Return (color_name, text)."""
    # dist_trend: negative => approaching
    if dist <= VENUS_HIT_RADIUS * 1.4:
        return ("green", "かなり近い！そのまま！")
    if dist_trend < -1e-4:
        return ("green", "近づいてる（いい！）")
    if abs(dist_trend) <= 1e-4:
        return ("yellow", "あまり変わらない（βを変える？）")
    return ("red", "遠ざかってる（βの符号を疑え）")


# -----------------------------
# Streamlit app
# -----------------------------
st.set_page_config(page_title="IKAROS β-GO!", layout="wide")
st.title("☀️ IKAROS β-GO!（2D・角度1本）")
st.caption("軌道は自動で進む。あなたは“帆のβ角”だけで、地球から金星へ近づけ！")

# Controls (top)
c1, c2, c3 = st.columns([1.2, 1.0, 1.1])

with c1:
    mode = st.radio("モード", ["リアルタイム制御", "事前角度指定"], horizontal=True)

with c2:
    time_unlimited = st.toggle("エクストラ：時間無制限", value=False)
    show_teacher = st.toggle("先生モード（詳細）", value=False)

with c3:
    st.markdown("**勝ち**：金星までの距離が小さくなるほど良い。十分近づいたらクリア！")
    st.markdown(f"- クリア距離：{VENUS_HIT_RADIUS:.3f} AU（おおよそ）")

# Config
dt_days = DT_DAYS_DEFAULT
a0 = A0_DEFAULT
venus_phase_deg = 60.0
time_limit_days = 220.0

if show_teacher:
    with st.expander("先生モード：難易度調整", expanded=False):
        dt_days = st.slider("刻み（days）", 0.2, 2.0, float(dt_days), 0.1)
        a0 = st.slider("帆の強さ（toy）", 0.005, 0.08, float(a0), 0.001)
        venus_phase_deg = st.slider("金星の初期位相（deg）", 0.0, 180.0, float(venus_phase_deg), 1.0)
        time_limit_days = st.slider("制限時間（days）", 60.0, 500.0, float(time_limit_days), 5.0)

cfg = GameConfig(
    dt_days=float(dt_days),
    a0=float(a0),
    venus_phase_deg=float(venus_phase_deg),
    time_limit_days=float(time_limit_days),
    time_unlimited=bool(time_unlimited),
)

# Session init/reset
if "game_cfg" not in st.session_state:
    st.session_state.game_cfg = cfg
    st.session_state.game = init_game(cfg)
    st.session_state.telemetry = []
    st.session_state.prev_dist = None
else:
    # If cfg changed materially, keep current run unless user resets (teacher can tweak mid-run)
    st.session_state.game_cfg = cfg

# Buttons row
bcol1, bcol2, bcol3, bcol4, bcol5 = st.columns([1, 1, 1, 1, 1])

def reset_run():
    st.session_state.game = init_game(cfg)
    st.session_state.telemetry = []
    st.session_state.prev_dist = None

with bcol1:
    if st.button("🔁 リセット", use_container_width=True):
        reset_run()

game: GameState = st.session_state.game

# Beta adjust controls
def bump_beta(delta: float):
    game.beta_deg = float(clamp(game.beta_deg + delta, -75.0, 75.0))

with bcol2:
    if st.button("⬅ β -5°", use_container_width=True):
        bump_beta(-5.0)
with bcol3:
    if st.button("➡ β +5°", use_container_width=True):
        bump_beta(+5.0)
with bcol4:
    if st.button("⏸/▶ 自動進行", use_container_width=True):
        game.auto = not game.auto

with bcol5:
    # quick presets
    preset = st.selectbox("ワンタッチβ", ["-35°（内側へ）", "0°（待ち）", "+35°（外側へ）"], index=0)
    if preset.startswith("-35"):
        game.beta_deg = -35.0
    elif preset.startswith("0"):
        game.beta_deg = 0.0
    else:
        game.beta_deg = 35.0

# Show current beta big
st.markdown(
    f"## β = **{game.beta_deg:+.0f}°**  —  {beta_label(game.beta_deg)}"
)

# Pre-plan mode UI
plan: List[Tuple[float, float]] = []  # (beta_deg, duration_days)
run_plan = False
if mode == "事前角度指定":
    st.info("βを3区間だけ決めて、一気に実行します（HTV GO!!の“事前計画”っぽい遊び）。")
    p1, p2, p3 = st.columns(3)
    with p1:
        b1 = st.slider("区間1 β（deg）", -75, 75, -35, 1)
        d1 = st.slider("区間1 日数", 10, 200, 70, 1)
    with p2:
        b2 = st.slider("区間2 β（deg）", -75, 75, 0, 1)
        d2 = st.slider("区間2 日数", 0, 200, 30, 1)
    with p3:
        b3 = st.slider("区間3 β（deg）", -75, 75, -20, 1)
        d3 = st.slider("区間3 日数", 0, 250, 90, 1)
    plan = [(float(b1), float(d1)), (float(b2), float(d2)), (float(b3), float(d3))]
    run_plan = st.button("⏩ 計画を実行（最初から）", use_container_width=True)

# Simulation step function
def step_n(n_steps: int, beta_deg: float):
    if game.done:
        return
    dt = (cfg.dt_days / 365.25)

    for _ in range(n_steps):
        # time limit
        days = game.t_yr * 365.25
        if (not cfg.time_unlimited) and (days >= cfg.time_limit_days):
            game.done = True
            game.message = "時間切れ！でもエクストラ（時間無制限）で続けられる。"
            break

        # integrate spacecraft
        game.sc = rk4_step(game.sc, beta_deg=beta_deg, a0=cfg.a0, dt=dt)
        game.t_yr += dt

        # telemetry (downsample to keep light)
        if len(st.session_state.telemetry) < 6000:
            st.session_state.telemetry.append((game.t_yr, game.sc.copy(), beta_deg))

        # check win
        r_sc = game.sc[:2]
        r_v = planet_pos(R_V, N_V, game.t_yr, rad(cfg.venus_phase_deg))
        dist = norm(r_sc - r_v)
        game.best_dist = min(game.best_dist, dist)
        if dist <= VENUS_HIT_RADIUS:
            game.done = True
            game.message = "金星に十分近づいた！ミッション成功！"
            break

# Execute plan
if run_plan:
    reset_run()
    # Run through the plan
    for (b, d_days) in plan:
        if d_days <= 0:
            continue
        steps = int(max(1, round(d_days / cfg.dt_days)))
        step_n(steps, beta_deg=b)
        if game.done:
            break

# Real-time auto advance (or manual nudge)
# In real-time mode, 'auto' advances using autorefresh.
tick_steps = 3  # per refresh
if mode == "リアルタイム制御":
    if game.auto and (not game.done):
        # autorefresh drives reruns
        st_autorefresh = st.experimental_data_editor  # placeholder to avoid import confusion (not used)

        # Streamlit's native autorefresh is in st_autorefresh (streamlit>=1.25+).
        # We'll call it if available, else fall back to manual stepping.
        try:
            from streamlit_autorefresh import st_autorefresh as _st_autorefresh  # optional package
            _st_autorefresh(interval=350, key="tick")
            step_n(tick_steps, beta_deg=game.beta_deg)
        except Exception:
            # No extra dependency; use built-in st.experimental_rerun with a soft hint:
            # We do a small step per run when user presses buttons.
            pass

# Manual single nudge (works even without autorefresh)
if mode == "リアルタイム制御":
    nudge = st.button("▶ ちょっと進める（約3日）", use_container_width=True)
    if nudge and (not game.done):
        steps = int(max(1, round(3.0 / cfg.dt_days)))
        step_n(steps, beta_deg=game.beta_deg)

# Compute metrics for UI
t_days = game.t_yr * 365.25
r_sc = game.sc[:2]
r_e = planet_pos(R_E, N_E, game.t_yr, 0.0)
r_v = planet_pos(R_V, N_V, game.t_yr, rad(cfg.venus_phase_deg))
dist_v = norm(r_sc - r_v)

prev = st.session_state.prev_dist
dist_trend = 0.0 if prev is None else (dist_v - prev)
st.session_state.prev_dist = dist_v

color, advice = traffic_light(dist_trend, dist_v)

# Top HUD
h1, h2, h3, h4 = st.columns(4)
h1.metric("経過日数", f"{t_days:.0f} d")
if cfg.time_unlimited:
    h2.metric("残り時間", "∞")
else:
    h2.metric("残り時間", f"{max(0.0, cfg.time_limit_days - t_days):.0f} d")
h3.metric("金星まで距離", f"{dist_v:.3f} AU")
h4.metric("ベスト距離", f"{game.best_dist:.3f} AU")

# Traffic light render (simple, color-free for robustness)
if color == "green":
    st.success(f"● 緑  {advice}")
elif color == "yellow":
    st.warning(f"● 黄  {advice}")
else:
    st.error(f"● 赤  {advice}")

# Main plot
left, right = st.columns([1.35, 0.9], gap="large")

with left:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (AU)")
    ax.set_ylabel("y (AU)")

    # Orbits (circles)
    th = np.linspace(0, 2*math.pi, 400)
    ax.plot(R_E * np.cos(th), R_E * np.sin(th), linewidth=1)
    ax.plot(R_V * np.cos(th), R_V * np.sin(th), linewidth=1)

    # Trajectory
    if st.session_state.telemetry:
        traj = np.array([s[:2] for (_, s, _) in st.session_state.telemetry], dtype=float)
        ax.plot(traj[:, 0], traj[:, 1], linewidth=2)

    # Bodies
    ax.scatter([0.0], [0.0], marker="o")  # Sun
    ax.text(0.02, 0.02, "Sun", fontsize=10)

    ax.scatter([r_e[0]], [r_e[1]], marker="o")
    ax.text(r_e[0] + 0.02, r_e[1] + 0.02, "Earth", fontsize=10)

    ax.scatter([r_v[0]], [r_v[1]], marker="o")
    ax.text(r_v[0] + 0.02, r_v[1] + 0.02, "Venus", fontsize=10)

    ax.scatter([r_sc[0]], [r_sc[1]], marker="x")
    ax.text(r_sc[0] + 0.02, r_sc[1] + 0.02, "Sail", fontsize=10)

    # Sail acceleration arrow (hint)
    a_vec = sail_accel(r_sc, game.beta_deg, cfg.a0)
    a_hat = unit(a_vec)
    ax.arrow(r_sc[0], r_sc[1], 0.12 * a_hat[0], 0.12 * a_hat[1], head_width=0.03, length_includes_head=True)

    # View limits
    lim = 1.35
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    st.pyplot(fig, use_container_width=True)

with right:
    st.subheader("遊び方（これだけ）")
    st.markdown(
        """
- **β<0** にすると、進行方向と反対に押して **減速** → 内側（＝金星側）へ行きやすい  
- **β=0** は “待ち”（大きく変えたくない時）  
- **β>0** は **加速** → 外側へ行きやすい  
"""
    )
    st.markdown("**コツ（最短の説明）**")
    st.info("まず β=-35° で内側へ。すれ違いそうなら β=0° で“待ち”を入れてタイミング調整。")

    st.divider()
    if game.done:
        if "成功" in game.message:
            st.success(game.message)
            st.balloons()
        else:
            st.error(game.message)

    st.caption("※ 本モデルは教材用の簡略化です（実機の精密な航法・姿勢・光圧モデルではありません）。")

    if show_teacher and st.session_state.telemetry:
        with st.expander("先生モード：簡易ログ（最後の200点）", expanded=False):
            rows = []
            for (t, s, b) in st.session_state.telemetry[-200:]:
                rv = planet_pos(R_V, N_V, t, rad(cfg.venus_phase_deg))
                rows.append(
                    dict(
                        day=t*365.25,
                        x=float(s[0]), y=float(s[1]),
                        vx=float(s[2]), vy=float(s[3]),
                        beta=float(b),
                        dist_venus=float(norm(s[:2]-rv)),
                    )
                )
            st.dataframe(rows, use_container_width=True)

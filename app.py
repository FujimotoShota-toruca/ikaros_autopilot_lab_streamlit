# IKAROS β-GO! Venus Corridor (Streamlit)
# ------------------------------------------------------------
# Educational toy focusing on "small continuous acceleration" guidance:
# - Baseline is a Venus-bound transfer-like orbit (Hohmann-ish) so the story is realistic:
#   "we are already mostly on the way to Venus."
# - Player controls ONE parameter: β (signed cone angle in 2D).
# - Goal is NOT "huge orbit change", but "meet Venus under constraints":
#   - Approach corridor: achieve a local-minimum distance within [d_min, d_max].
#   - Ops constraint: keep power available during comm windows (penalty if violated).
#
# Modes:
# 1) リアルタイム制御：βを変えながら誘導（自動進行/一時停止）
# 2) 事前角度指定：βを3区間だけ先に決めて一気に実行
#
# Propulsion:
# - SRP (solar radiation pressure only): tiny ΔV (tens of m/s over months)
# - SPS(IES) (solar power sail + ion engine assist): larger effective ΔV
#   (still small vs orbital speed, but noticeably more authority).
#
# Extra:
# - 時間無制限（練習）
#
# Note: This is a simplified model for teaching. Not mission design software.
# ------------------------------------------------------------

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

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

def sign(x: float) -> float:
    if x > 1e-12:
        return 1.0
    if x < -1e-12:
        return -1.0
    return 0.0


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

# Unit conversion (for "ΔV相当" display)
AU = 1.495978707e11
YR = 365.25 * 24 * 3600
AUYR2_TO_MPS2 = AU / (YR**2)

# Win condition: approach corridor (toy)
D_MIN = 0.012  # AU
D_MAX = 0.030  # AU


# -----------------------------
# Baseline transfer (Hohmann-ish) setup
# Start at r=1 AU (aphelion) heading inward to r=0.723 AU (perihelion)
# -----------------------------
A_H = 0.5 * (R_E + R_V)  # semi-major axis
T_HALF_YR = 0.5 * math.sqrt(A_H**3)  # years (because MU=4π²)
T_HALF_DAYS = T_HALF_YR * 365.25

def hohmann_aphelion_speed() -> float:
    # v = sqrt(mu*(2/r - 1/a))
    r = R_E
    a = A_H
    return math.sqrt(MU * (2.0 / r - 1.0 / a))

def circular_speed(r: float) -> float:
    return math.sqrt(MU / r)

V_CIRC_E = circular_speed(R_E)
V_APH = hohmann_aphelion_speed()


# -----------------------------
# Sail + IES model (single control β)
# -----------------------------
def power_available(beta_deg: float) -> float:
    # Treat solar cell power ~ cos(incidence) = cos(|β|)
    b = abs(rad(beta_deg))
    return clamp(math.cos(b), 0.0, 1.0)

def srp_accel(r_sc: np.ndarray, beta_deg: float, a0_srp: float) -> np.ndarray:
    """SRP-only: magnitude ∝ (1/r^2) * cos^2(beta), direction along sail normal in-plane."""
    r = norm(r_sc)
    beta = rad(beta_deg)
    r_hat = unit(r_sc)
    t_hat = rot90(r_hat)

    c = math.cos(beta)
    s = math.sin(beta)
    n_hat = c * r_hat + s * t_hat

    mag = a0_srp * (c * c) / max(r*r, 1e-12)
    return mag * n_hat

def ies_accel(r_sc: np.ndarray, beta_deg: float, a0_ies: float) -> np.ndarray:
    """Toy IES assist: along-track thrust, limited by available power and 1/r^2."""
    r = norm(r_sc)
    r_hat = unit(r_sc)
    t_hat = rot90(r_hat)  # prograde

    p = power_available(beta_deg)
    sgn = sign(beta_deg)
    if sgn == 0.0:
        return np.zeros(2, dtype=float)

    mag = a0_ies * p / max(r*r, 1e-12)
    return (sgn * mag) * t_hat

def total_accel(r_sc: np.ndarray, beta_deg: float, a0_srp: float, a0_ies: float, mode: str) -> np.ndarray:
    a = srp_accel(r_sc, beta_deg, a0_srp)
    if mode == "SPS(IES)":
        a = a + ies_accel(r_sc, beta_deg, a0_ies)
    return a


# -----------------------------
# Dynamics integration (RK4)
# -----------------------------
def grav_accel(r: np.ndarray) -> np.ndarray:
    d = norm(r)
    return -MU * r / (d**3 + 1e-12)

def f(state: np.ndarray, beta_deg: float, a0_srp: float, a0_ies: float, prop_mode: str) -> np.ndarray:
    # state = [x,y,vx,vy]
    r = state[:2]
    v = state[2:]
    a = grav_accel(r) + total_accel(r, beta_deg, a0_srp, a0_ies, prop_mode)
    return np.array([v[0], v[1], a[0], a[1]], dtype=float)

def rk4_step(state: np.ndarray, beta_deg: float, a0_srp: float, a0_ies: float, prop_mode: str, dt: float) -> np.ndarray:
    k1 = f(state, beta_deg, a0_srp, a0_ies, prop_mode)
    k2 = f(state + 0.5 * dt * k1, beta_deg, a0_srp, a0_ies, prop_mode)
    k3 = f(state + 0.5 * dt * k2, beta_deg, a0_srp, a0_ies, prop_mode)
    k4 = f(state + dt * k3, beta_deg, a0_srp, a0_ies, prop_mode)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# -----------------------------
# Planets (analytic circles)
# -----------------------------
def planet_pos(r: float, n: float, t_yr: float, phase_rad: float) -> np.ndarray:
    ang = n * t_yr + phase_rad
    return np.array([r * math.cos(ang), r * math.sin(ang)], dtype=float)

def earth_pos(t_yr: float) -> np.ndarray:
    return planet_pos(R_E, N_E, t_yr, 0.0)

def venus_pos(t_yr: float, phase_rad: float) -> np.ndarray:
    return planet_pos(R_V, N_V, t_yr, phase_rad)

def hohmann_aligned_venus_phase() -> float:
    """Choose Venus initial phase so that a Hohmann-ish transfer arriving at ~T_HALF hits Venus near opposition."""
    # spacecraft goes ~pi radians in time T_HALF; want Venus angle = pi at t=T_HALF
    return (math.pi - N_V * T_HALF_YR) % (2 * math.pi)


# -----------------------------
# Ops constraint: comm windows
# -----------------------------
def comm_window_active(t_days: float, period: float = 25.0, duration: float = 3.0) -> bool:
    # every 'period' days, first 'duration' days are comm window
    x = t_days % period
    return x < duration


# -----------------------------
# Game state
# -----------------------------
@dataclass
class GameConfig:
    dt_days: float
    time_limit_days: float
    time_unlimited: bool
    prop_mode: str
    a0_srp: float
    a0_ies: float
    venus_phase_deg: float  # initial phase

@dataclass
class ApproachEvent:
    t_days: float
    dist: float
    rel_v: float

@dataclass
class GameState:
    t_yr: float
    sc: np.ndarray  # [x,y,vx,vy]
    beta_deg: float
    auto: bool
    done: bool
    message: str
    best_dist: float
    dv_mps: float
    comm_viol_s: float
    last_dist: Optional[float]
    dist_trend: float
    # For local minima detection:
    prev2_dist: Optional[float]
    prev_dist: Optional[float]
    minima: List[ApproachEvent]


def init_game(cfg: GameConfig) -> GameState:
    # Start on transfer-like orbit at Earth's position (x=1, y=0) with aphelion speed V_APH prograde.
    r0 = np.array([R_E, 0.0], dtype=float)
    v0 = np.array([0.0, V_APH], dtype=float)  # prograde (CCW), slower than circular -> falls inward
    sc = np.array([r0[0], r0[1], v0[0], v0[1]], dtype=float)

    return GameState(
        t_yr=0.0,
        sc=sc,
        beta_deg=-20.0,
        auto=True,
        done=False,
        message="",
        best_dist=1e9,
        dv_mps=0.0,
        comm_viol_s=0.0,
        last_dist=None,
        dist_trend=0.0,
        prev2_dist=None,
        prev_dist=None,
        minima=[],
    )


# -----------------------------
# UI helpers
# -----------------------------
def beta_label(beta_deg: float) -> str:
    if beta_deg < -1e-6:
        return "減速（内側へ）"
    if beta_deg > 1e-6:
        return "加速（外側へ）"
    return "待ち（変えない）"

def corridor_status(dmin: float, dmax: float, x: float) -> Tuple[str, str]:
    if dmin <= x <= dmax:
        return ("green", f"回廊OK（{dmin:.3f}〜{dmax:.3f} AU）")
    if x < dmin:
        return ("red", f"近すぎ（< {dmin:.3f} AU）")
    return ("yellow", f"遠い（> {dmax:.3f} AU）")


# -----------------------------
# Streamlit app
# -----------------------------
st.set_page_config(page_title="IKAROS β-GO! Venus Corridor", layout="wide")

# Keep the page short so β is "always visible"
st.markdown(
    """
<style>
.block-container {padding-top: 1.0rem; padding-bottom: 1.0rem;}
</style>
""",
    unsafe_allow_html=True,
)

st.title("☀️ IKAROS β-GO!（金星回廊ミッション）")
st.caption("“もう金星へ向かう軌道”に乗っている前提で、β角の微調整で『狙い通りに近づく』をやる。")

# Top controls (single row)
col1, col2, col3, col4 = st.columns([1.25, 1.05, 1.05, 1.15], gap="medium")
with col1:
    mode = st.radio("モード", ["リアルタイム制御", "事前角度指定"], horizontal=True)
with col2:
    prop_mode = st.radio("推進", ["SRP", "SPS(IES)"], horizontal=True, help="SRPは太陽光圧のみ。SPS(IES)は電気推進の併用（ソーラー電力セイルのイメージ）。")
with col3:
    time_unlimited = st.toggle("エクストラ：時間無制限", value=False)
    show_teacher = st.toggle("先生モード", value=False)
with col4:
    st.markdown("**目標**")
    st.markdown(f"- 金星最近接を回廊に：**{D_MIN:.3f}〜{D_MAX:.3f} AU**")
    st.markdown("- 通信ウィンドウでは『発電（β小さめ）』が必要")

# Defaults (teacher can tweak)
dt_days = 0.5
time_limit_days = 220.0

# SRP magnitude near IKAROS-scale (toy units)
a0_srp = 0.025  # AU/yr^2 at 1AU when beta=0 (then scaled by cos^2 and 1/r^2)
a0_ies = 0.12   # AU/yr^2 (toy) along-track assist when enabled

# Venus phase aligned to baseline transfer
venus_phase_deg = deg(hohmann_aligned_venus_phase())

if show_teacher:
    with st.expander("先生モード：難易度（いじりすぎ注意）", expanded=False):
        dt_days = st.slider("刻み（days）", 0.2, 2.0, float(dt_days), 0.1)
        time_limit_days = st.slider("制限時間（days）", 80.0, 500.0, float(time_limit_days), 5.0)
        a0_srp = st.slider("SRP強さ（AU/yr^2）", 0.002, 0.06, float(a0_srp), 0.001)
        a0_ies = st.slider("IES強さ（AU/yr^2）", 0.02, 0.35, float(a0_ies), 0.01)
        venus_phase_deg = st.slider("金星 初期位相（deg）", 0.0, 360.0, float(venus_phase_deg), 1.0)

cfg = GameConfig(
    dt_days=float(dt_days),
    time_limit_days=float(time_limit_days),
    time_unlimited=bool(time_unlimited),
    prop_mode=str(prop_mode),
    a0_srp=float(a0_srp),
    a0_ies=float(a0_ies),
    venus_phase_deg=float(venus_phase_deg),
)

# Session init/reset
if "vcfg" not in st.session_state:
    st.session_state.vcfg = cfg
    st.session_state.game = init_game(cfg)
    st.session_state.telemetry = []  # (t_yr, sc_state, beta_deg)
else:
    st.session_state.vcfg = cfg

game: GameState = st.session_state.game

# Buttons row
b1, b2, b3, b4, b5, b6 = st.columns([1.0, 1.0, 1.0, 1.0, 1.1, 1.2], gap="small")

def reset_run():
    st.session_state.game = init_game(cfg)
    st.session_state.telemetry = []
    st.session_state.game_cfg_snapshot = cfg

with b1:
    if st.button("🔁 リセット", use_container_width=True):
        reset_run()
        game = st.session_state.game

def bump_beta(delta: float):
    game.beta_deg = float(clamp(game.beta_deg + delta, -75.0, 75.0))

with b2:
    if st.button("⬅ β -5°", use_container_width=True):
        bump_beta(-5.0)
with b3:
    if st.button("➡ β +5°", use_container_width=True):
        bump_beta(+5.0)
with b4:
    if st.button("⏸/▶ 自動進行", use_container_width=True):
        game.auto = not game.auto

with b5:
    preset = st.selectbox("ワンタッチβ", ["-35°（内側へ）", "0°（待ち）", "+35°（外側へ）"], index=0, label_visibility="visible")
    if preset.startswith("-35"):
        game.beta_deg = -35.0
    elif preset.startswith("0"):
        game.beta_deg = 0.0
    else:
        game.beta_deg = 35.0

with b6:
    nudge = st.button("▶ ちょっと進める（約5日）", use_container_width=True)

# Always-visible HUD row (β included)
t_days = game.t_yr * 365.25
r_sc = game.sc[:2]
v_sc = game.sc[2:]
venus_phase_rad = rad(cfg.venus_phase_deg)
r_v = venus_pos(game.t_yr, venus_phase_rad)
v_v = np.array([-R_V * N_V * math.sin(N_V * game.t_yr + venus_phase_rad), R_V * N_V * math.cos(N_V * game.t_yr + venus_phase_rad)], dtype=float)

dist_v = norm(r_sc - r_v)
rel_v = norm(v_sc - v_v)  # AU/yr

# Local minima detection (on current dist, using last two)
if game.prev2_dist is None:
    game.prev2_dist = dist_v
    game.prev_dist = dist_v
else:
    pass
# comm window
comms = comm_window_active(t_days)
pwr = power_available(game.beta_deg)

# Corridor on best local minima so far
best_min = min([ev.dist for ev in game.minima], default=game.best_dist)
colH1, colH2, colH3, colH4, colH5, colH6 = st.columns([1,1,1,1,1,1], gap="small")
colH1.metric("β角", f"{game.beta_deg:+.0f}°", delta=beta_label(game.beta_deg))
colH2.metric("経過", f"{t_days:.0f} d", delta=f"目安: {T_HALF_DAYS:.0f} d で接近")
colH3.metric("金星距離", f"{dist_v:.3f} AU")
colH4.metric("相対速度", f"{rel_v:.2f} AU/yr")
colH5.metric("発電", f"{pwr:.2f}", delta=("COMMS" if comms else ""))
colH6.metric("ΔV相当", f"{game.dv_mps:.0f} m/s", help="加速度を積分した参考値（おおよそ）。")

# Status banners
corr_color, corr_text = corridor_status(D_MIN, D_MAX, best_min)
if corr_color == "green":
    st.success(f"回廊ステータス：{corr_text}（これまでの最近接 {best_min:.3f} AU）")
elif corr_color == "yellow":
    st.warning(f"回廊ステータス：{corr_text}（これまでの最近接 {best_min:.3f} AU）")
else:
    st.error(f"回廊ステータス：{corr_text}（これまでの最近接 {best_min:.3f} AU）")

# Pre-plan mode UI
plan: List[Tuple[float, float]] = []  # (beta_deg, duration_days)
run_plan = False
if mode == "事前角度指定":
    st.info("βを3区間だけ決めて一気に実行：『基本は金星遷移、βは微調整』が分かりやすい。")
    p1, p2, p3 = st.columns(3)
    with p1:
        b1p = st.slider("区間1 β（deg）", -75, 75, -35, 1)
        d1p = st.slider("区間1 日数", 10, 220, 90, 1)
    with p2:
        b2p = st.slider("区間2 β（deg）", -75, 75, 0, 1)
        d2p = st.slider("区間2 日数", 0, 200, 30, 1)
    with p3:
        b3p = st.slider("区間3 β（deg）", -75, 75, -15, 1)
        d3p = st.slider("区間3 日数", 0, 260, 120, 1)
    plan = [(float(b1p), float(d1p)), (float(b2p), float(d2p)), (float(b3p), float(d3p))]
    run_plan = st.button("⏩ 計画を実行（最初から）", use_container_width=True)


# -----------------------------
# Simulation step
# -----------------------------
def update_dv_and_comm(beta_deg: float, dt_yr: float):
    # ΔV ~ ∫|a_prop| dt (toy)
    r = st.session_state.game.sc[:2]
    a = total_accel(r, beta_deg, cfg.a0_srp, cfg.a0_ies, cfg.prop_mode)  # AU/yr^2
    a_mps2 = norm(a) * AUYR2_TO_MPS2
    dt_s = dt_yr * YR
    st.session_state.game.dv_mps += a_mps2 * dt_s

    # comm constraint: in comm window, require high-ish power (beta small)
    t_days_now = st.session_state.game.t_yr * 365.25
    if comm_window_active(t_days_now):
        if power_available(beta_deg) < 0.70:  # threshold
            st.session_state.game.comm_viol_s += dt_s

def record_and_check_minimum(dist_now: float, rel_v_now: float):
    g = st.session_state.game
    if g.prev2_dist is None or g.prev_dist is None:
        g.prev2_dist = dist_now
        g.prev_dist = dist_now
        return

    # detect local minimum at prev_dist (prev2 > prev < now)
    if (g.prev_dist < g.prev2_dist) and (g.prev_dist < dist_now):
        # local minimum occurred at previous step
        t_days_min = (g.t_yr - (cfg.dt_days/365.25)) * 365.25
        ev = ApproachEvent(t_days=t_days_min, dist=float(g.prev_dist), rel_v=float(rel_v_now))
        g.minima.append(ev)

        # Success if within corridor
        if D_MIN <= ev.dist <= D_MAX and (not g.done):
            g.done = True
            g.message = f"回廊に入った！最近接 {ev.dist:.3f} AU（{ev.t_days:.0f}日）"
            return

    g.prev2_dist = g.prev_dist
    g.prev_dist = dist_now

def step_n(n_steps: int, beta_deg: float):
    g = st.session_state.game
    if g.done:
        return

    dt_yr = (cfg.dt_days / 365.25)

    for _ in range(n_steps):
        # time limit
        days = g.t_yr * 365.25
        if (not cfg.time_unlimited) and (days >= cfg.time_limit_days):
            g.done = True
            g.message = "時間切れ！エクストラ（時間無制限）で続けて練習できる。"
            break

        # integrate
        g.sc = rk4_step(g.sc, beta_deg=beta_deg, a0_srp=cfg.a0_srp, a0_ies=cfg.a0_ies, prop_mode=cfg.prop_mode, dt=dt_yr)
        g.t_yr += dt_yr

        # Track dv + comm penalty
        update_dv_and_comm(beta_deg, dt_yr)

        # telemetry downsample
        if len(st.session_state.telemetry) < 8000:
            st.session_state.telemetry.append((g.t_yr, g.sc.copy(), beta_deg))

        # update distance and minima
        r_sc2 = g.sc[:2]
        v_sc2 = g.sc[2:]
        r_v2 = venus_pos(g.t_yr, venus_phase_rad)
        v_v2 = np.array([-R_V * N_V * math.sin(N_V * g.t_yr + venus_phase_rad), R_V * N_V * math.cos(N_V * g.t_yr + venus_phase_rad)], dtype=float)
        dist2 = norm(r_sc2 - r_v2)
        relv2 = norm(v_sc2 - v_v2)
        g.best_dist = min(g.best_dist, dist2)
        record_and_check_minimum(dist2, relv2)

        # stop if success
        if g.done and "回廊に入った" in g.message:
            break


# Fix small typo: dt variable name
# (We patch by defining dt_yr alias below and rewriting the erroneous line if it exists)

# Execute plan
if run_plan:
    reset_run()
    # Run through the plan
    for (b, d_days) in plan:
        if d_days <= 0:
            continue
        steps = int(max(1, round(d_days / cfg.dt_days)))
        step_n(steps, beta_deg=b)
        if st.session_state.game.done:
            break

# Realtime auto advance
tick_steps = 2  # steps per refresh
if mode == "リアルタイム制御":
    if game.auto and (not game.done):
        try:
            from streamlit_autorefresh import st_autorefresh as _st_autorefresh  # type: ignore
            _st_autorefresh(interval=350, key="tick")
            step_n(tick_steps, beta_deg=game.beta_deg)
        except Exception:
            # If component isn't available, users can still use the nudge button.
            pass

if mode == "リアルタイム制御" and nudge and (not game.done):
    steps = int(max(1, round(5.0 / cfg.dt_days)))
    step_n(steps, beta_deg=game.beta_deg)

# Main plot + right panel
left, right = st.columns([1.35, 0.95], gap="large")

with left:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (AU)")
    ax.set_ylabel("y (AU)")

    # Orbits
    th = np.linspace(0, 2*math.pi, 400)
    ax.plot(R_E * np.cos(th), R_E * np.sin(th), linewidth=1)
    ax.plot(R_V * np.cos(th), R_V * np.sin(th), linewidth=1)

    # Trajectory
    if st.session_state.telemetry:
        traj = np.array([s[:2] for (_, s, _) in st.session_state.telemetry], dtype=float)
        ax.plot(traj[:, 0], traj[:, 1], linewidth=2)

    # Bodies
    ax.scatter([0.0], [0.0], marker="o")
    ax.text(0.02, 0.02, "Sun", fontsize=10)

    r_e = earth_pos(st.session_state.game.t_yr)
    ax.scatter([r_e[0]], [r_e[1]], marker="o")
    ax.text(r_e[0] + 0.02, r_e[1] + 0.02, "Earth", fontsize=10)

    r_v_now = venus_pos(st.session_state.game.t_yr, venus_phase_rad)
    ax.scatter([r_v_now[0]], [r_v_now[1]], marker="o")
    ax.text(r_v_now[0] + 0.02, r_v_now[1] + 0.02, "Venus", fontsize=10)

    r_sc_now = st.session_state.game.sc[:2]
    ax.scatter([r_sc_now[0]], [r_sc_now[1]], marker="x")
    ax.text(r_sc_now[0] + 0.02, r_sc_now[1] + 0.02, f"Sail β={st.session_state.game.beta_deg:+.0f}°", fontsize=10)

    # Acceleration arrow
    a_vec = total_accel(r_sc_now, st.session_state.game.beta_deg, cfg.a0_srp, cfg.a0_ies, cfg.prop_mode)
    if norm(a_vec) > 1e-12:
        a_hat = unit(a_vec)
        ax.arrow(r_sc_now[0], r_sc_now[1], 0.12 * a_hat[0], 0.12 * a_hat[1], head_width=0.03, length_includes_head=True)

    # View limits
    lim = 1.35
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    st.pyplot(fig, use_container_width=True)

with right:
    st.subheader("遊び方（これだけ）")
    st.markdown(
        """
- **β<0**：進行方向と反対に押して減速 → 内側（＝金星側）へ行きやすい  
- **β=0**：待ち（タイミング合わせ）  
- **β>0**：加速 → 外側へ行きやすい  
"""
    )

    st.markdown("**今回のポイント**")
    st.info("太陽光圧は小さいので、“大改造”ではなく**早めの微調整**が効く。SPS(IES)に切り替えると、同じβでも効きが増える。")

    st.divider()
    if comms:
        if pwr < 0.70:
            st.error("通信ウィンドウ中：発電不足！（βを0°に寄せよう）")
        else:
            st.success("通信ウィンドウ中：OK（β小さめ）")
    else:
        st.caption("通信ウィンドウ：ときどき発電が必要（先生モードで周期調整は未対応）")

    st.divider()
    st.markdown("**最近接イベント（検出したもの）**")
    if st.session_state.game.minima:
        last = st.session_state.game.minima[-1]
        st.write(f"最新：{last.t_days:.0f}日  dist={last.dist:.3f} AU")
        with st.expander("一覧（最大10件）", expanded=False):
            for ev in st.session_state.game.minima[-10:]:
                st.write(f"- {ev.t_days:.0f}日  dist={ev.dist:.3f} AU")
    else:
        st.caption("まだ最近接イベントがありません。時間を進めると出ます。")

    st.divider()
    if st.session_state.game.done:
        if "回廊に入った" in st.session_state.game.message:
            st.success(st.session_state.game.message)
            st.balloons()
        else:
            st.warning(st.session_state.game.message)

    if show_teacher:
        st.divider()
        st.subheader("先生メモ")
        st.caption("ΔV相当（参考値）と通信違反時間（秒）を表示します。")
        st.write(f"- ΔV相当: {st.session_state.game.dv_mps:.0f} m/s")
        st.write(f"- 通信違反: {st.session_state.game.comm_viol_s/3600:.1f} 時間")

st.caption("※ 教材用の簡略モデルです（実機の精密な航法・姿勢・光圧モデルではありません）。")

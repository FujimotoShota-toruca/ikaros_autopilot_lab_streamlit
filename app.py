# IKAROS β-GO! (HTV-GO style) - SRP only
# ------------------------------------------------------------
# Requested features:
# - Nominal (target) trajectory shown (dashed).
# - Prediction "if keep current β" shown (dotted).
# - Live displays: power, Earth angle, orbit error time-series, fuel consumption.
# - After time limit: result screen with graphs + score.
# - Remove IES mode and pre-planned mode (SRP only).
# - Optional high difficulty: β can be changed only during comm window, fuel limited.
#
# Educational simplifications:
# - 2D heliocentric dynamics, planets on circular orbits.
# - "Fuel" is a stand-in for attitude-control resource (e.g., gas/actuator budget),
#   consumed when changing β.
# - SRP acceleration is scaled for visibility but still yields small ΔV vs orbital speed.
# ------------------------------------------------------------

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Optional autorefresh
try:
    from streamlit_autorefresh import st_autorefresh
    HAVE_AUTOREFRESH = True
except Exception:
    HAVE_AUTOREFRESH = False


# -----------------------------
# Helpers
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
    return np.array([-v[1], v[0]], dtype=float)

def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    uu = unit(u)
    vv = unit(v)
    c = float(np.clip(np.dot(uu, vv), -1.0, 1.0))
    return math.acos(c)


# -----------------------------
# Toy solar system (AU, yr)
# -----------------------------
MU = 4.0 * math.pi * math.pi  # AU^3/yr^2

R_E = 1.0
R_V = 0.723

def mean_motion(r: float) -> float:
    return math.sqrt(MU / (r**3))

N_E = mean_motion(R_E)
N_V = mean_motion(R_V)

# unit conversion for display
AU_M = 1.495978707e11
YR_S = 365.25 * 24 * 3600.0
AUYR2_TO_MPS2 = AU_M / (YR_S**2)


def planet_pos(r: float, n: float, t_yr: float, phase: float) -> np.ndarray:
    a = n * t_yr + phase
    return np.array([r * math.cos(a), r * math.sin(a)], dtype=float)

def planet_vel(r: float, n: float, t_yr: float, phase: float) -> np.ndarray:
    a = n * t_yr + phase
    # derivative of [r cos, r sin] is [-r n sin, r n cos]
    return np.array([-r * n * math.sin(a), r * n * math.cos(a)], dtype=float)


# -----------------------------
# Baseline transfer-like start (Hohmann-ish)
# -----------------------------
A_H = 0.5 * (R_E + R_V)
T_HALF_YR = 0.5 * math.sqrt(A_H**3)  # because MU=4π²
T_HALF_DAYS = T_HALF_YR * 365.25

def circular_speed(r: float) -> float:
    return math.sqrt(MU / r)

def hohmann_aphelion_speed() -> float:
    r = R_E
    a = A_H
    return math.sqrt(MU * (2.0 / r - 1.0 / a))

V_APH = hohmann_aphelion_speed()
V_CIRC_E = circular_speed(R_E)

def aligned_venus_phase() -> float:
    # want Venus angle = pi at t=T_HALF (spacecraft roughly goes pi rad on transfer)
    return (math.pi - N_V * T_HALF_YR) % (2 * math.pi)


# -----------------------------
# SRP model (β only, in-plane)
# -----------------------------
def power_frac(r_sc: np.ndarray, beta_deg: float) -> float:
    # simple: power ∝ (1/r^2) * cos(|β|)
    r = max(norm(r_sc), 1e-9)
    return clamp(math.cos(abs(rad(beta_deg))) / (r*r), 0.0, 2.0)

def srp_accel(r_sc: np.ndarray, beta_deg: float, a0: float) -> np.ndarray:
    # magnitude ∝ (1/r^2) * cos^2(beta), direction along in-plane sail normal
    r = max(norm(r_sc), 1e-9)
    beta = rad(beta_deg)
    r_hat = unit(r_sc)
    t_hat = rot90(r_hat)

    c = math.cos(beta)
    s = math.sin(beta)
    n_hat = c * r_hat + s * t_hat

    mag = a0 * (c*c) / (r*r)
    return mag * n_hat


# -----------------------------
# Dynamics (RK4)
# -----------------------------
def grav_accel(r: np.ndarray) -> np.ndarray:
    d = max(norm(r), 1e-12)
    return -MU * r / (d**3)

def f(state: np.ndarray, beta_deg: float, a0: float) -> np.ndarray:
    r = state[:2]
    v = state[2:]
    a = grav_accel(r) + srp_accel(r, beta_deg, a0)
    return np.array([v[0], v[1], a[0], a[1]], dtype=float)

def rk4_step(state: np.ndarray, beta_deg: float, a0: float, dt: float) -> np.ndarray:
    k1 = f(state, beta_deg, a0)
    k2 = f(state + 0.5*dt*k1, beta_deg, a0)
    k3 = f(state + 0.5*dt*k2, beta_deg, a0)
    k4 = f(state + dt*k3, beta_deg, a0)
    return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


# -----------------------------
# Nominal plan (simple, fixed schedule) -> nominal trajectory
# You can tweak these to show "nominal vs actual".
# -----------------------------
def nominal_beta_schedule(t_days: float) -> float:
    # Simple "plan": early inward push, then wait, then small trim.
    if t_days < 70:
        return -20.0
    if t_days < 100:
        return 0.0
    return -10.0


# -----------------------------
# Game config/state
# -----------------------------
@dataclass
class Config:
    dt_days: float = 0.5
    ticks_per_refresh: int = 2
    time_limit_days: float = 220.0
    a0: float = 0.020  # AU/yr^2 at 1AU beta=0 (scaled by 1/r^2 and cos^2)
    venus_phase_offset_deg: float = 18.0  # offset from perfect Hohmann alignment
    comm_angle_deg: float = 12.0  # Earth angle threshold for comm window
    fuel_start: float = 120.0
    fuel_per_deg: float = 0.35  # fuel per degree change (attitude resource)
    hard_mode: bool = False

@dataclass
class SimState:
    t_yr: float
    x: np.ndarray  # [x,y,vx,vy]
    beta_deg: float
    beta_last_deg: float
    fuel: float
    dv_mps: float
    done: bool
    phase: str  # "play" or "result"
    message: str
    score: float
    # logs
    r_log: List[np.ndarray]
    t_days_log: List[float]
    beta_log: List[float]
    power_log: List[float]
    earth_angle_log: List[float]
    orbit_err_log: List[float]
    fuel_log: List[float]
    dv_log: List[float]
    # approach
    dist_venus_log: List[float]
    best_dist: float

def init_state(cfg: Config) -> SimState:
    # Start on transfer-like orbit at Earth's position with aphelion speed prograde (falls inward).
    r0 = np.array([R_E, 0.0], dtype=float)
    v0 = np.array([0.0, V_APH], dtype=float)
    x0 = np.array([r0[0], r0[1], v0[0], v0[1]], dtype=float)

    return SimState(
        t_yr=0.0,
        x=x0,
        beta_deg=-20.0,
        beta_last_deg=-20.0,
        fuel=cfg.fuel_start,
        dv_mps=0.0,
        done=False,
        phase="play",
        message="",
        score=0.0,
        r_log=[],
        t_days_log=[],
        beta_log=[],
        power_log=[],
        earth_angle_log=[],
        orbit_err_log=[],
        fuel_log=[],
        dv_log=[],
        dist_venus_log=[],
        best_dist=1e9,
    )


# -----------------------------
# Compute derived quantities
# -----------------------------
def earth_angle_deg(r_sc: np.ndarray, t_yr: float, earth_phase: float = 0.0) -> float:
    r_e = planet_pos(R_E, N_E, t_yr, earth_phase)
    # angle at spacecraft between direction to Sun and direction to Earth
    to_sun = -r_sc
    to_earth = r_e - r_sc
    return deg(angle_between(to_sun, to_earth))

def comm_window_active(r_sc: np.ndarray, t_yr: float, cfg: Config) -> bool:
    return earth_angle_deg(r_sc, t_yr) >= cfg.comm_angle_deg

def venus_phase(cfg: Config) -> float:
    return aligned_venus_phase() + rad(cfg.venus_phase_offset_deg)

def venus_distance(r_sc: np.ndarray, t_yr: float, cfg: Config) -> float:
    r_v = planet_pos(R_V, N_V, t_yr, venus_phase(cfg))
    return norm(r_sc - r_v)

def update_logs(state: SimState, cfg: Config, nominal_r: np.ndarray):
    t_days = state.t_yr * 365.25
    r_sc = state.x[:2]
    p = power_frac(r_sc, state.beta_deg)
    ea = earth_angle_deg(r_sc, state.t_yr)
    err = norm(r_sc - nominal_r)
    dV = state.dv_mps

    state.r_log.append(r_sc.copy())
    state.t_days_log.append(float(t_days))
    state.beta_log.append(float(state.beta_deg))
    state.power_log.append(float(p))
    state.earth_angle_log.append(float(ea))
    state.orbit_err_log.append(float(err))
    state.fuel_log.append(float(state.fuel))
    state.dv_log.append(float(dV))
    state.dist_venus_log.append(float(venus_distance(r_sc, state.t_yr, cfg)))
    state.best_dist = min(state.best_dist, state.dist_venus_log[-1])


# -----------------------------
# Nominal trajectory precompute
# -----------------------------
def simulate_nominal(cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return arrays (t_days, r_nom[N,2], beta_nom[N])."""
    dt_yr = cfg.dt_days / 365.25
    steps = int(math.ceil(cfg.time_limit_days / cfg.dt_days)) + 1

    # nominal starts from same initial state
    r0 = np.array([R_E, 0.0], dtype=float)
    v0 = np.array([0.0, V_APH], dtype=float)
    x = np.array([r0[0], r0[1], v0[0], v0[1]], dtype=float)

    t_yr = 0.0
    t_days = np.zeros(steps, dtype=float)
    r_nom = np.zeros((steps, 2), dtype=float)
    b_nom = np.zeros(steps, dtype=float)

    for i in range(steps):
        td = t_yr * 365.25
        b = nominal_beta_schedule(td)
        t_days[i] = td
        r_nom[i] = x[:2]
        b_nom[i] = b

        x = rk4_step(x, beta_deg=b, a0=cfg.a0, dt=dt_yr)
        t_yr += dt_yr

    return t_days, r_nom, b_nom


def predict_future(state: SimState, cfg: Config, horizon_days: float = 60.0) -> np.ndarray:
    """Predict future trajectory if β is held constant from current state."""
    dt_yr = cfg.dt_days / 365.25
    steps = int(max(2, round(horizon_days / cfg.dt_days)))
    x = state.x.copy()
    t_yr = state.t_yr
    traj = np.zeros((steps, 2), dtype=float)

    for i in range(steps):
        traj[i] = x[:2]
        x = rk4_step(x, beta_deg=state.beta_deg, a0=cfg.a0, dt=dt_yr)
        t_yr += dt_yr
    return traj


# -----------------------------
# Scoring
# -----------------------------
def compute_score(state: SimState, cfg: Config) -> Tuple[float, Dict[str, float]]:
    # Components (all toy)
    best = state.best_dist
    # approach: closer is better; scale ~ 0..500
    approach = 500.0 * math.exp(-best / 0.08)
    # tracking error penalty (integral)
    err_int = float(np.trapz(np.array(state.orbit_err_log), x=np.array(state.t_days_log))) if len(state.orbit_err_log) > 2 else 0.0
    track_pen = 1200.0 * err_int  # AU*day -> penalty
    # comm/power penalty: time below 0.75 power
    p = np.array(state.power_log) if state.power_log else np.array([1.0])
    t = np.array(state.t_days_log) if state.t_days_log else np.array([0.0])
    low = (p < 0.75).astype(float)
    low_time = float(np.trapz(low, x=t)) if len(t) > 2 else 0.0
    power_pen = 25.0 * low_time
    # fuel penalty
    fuel_used = cfg.fuel_start - state.fuel
    fuel_pen = 2.2 * max(0.0, fuel_used)
    # dv bonus (tiny; mostly for "SRP is small")
    dv_bonus = min(35.0, state.dv_mps * 0.08)

    score = approach + dv_bonus - track_pen - power_pen - fuel_pen
    score = max(0.0, score)

    breakdown = {
        "approach": approach,
        "dv_bonus": dv_bonus,
        "track_pen": track_pen,
        "power_pen": power_pen,
        "fuel_pen": fuel_pen,
        "best_dist_AU": best,
        "fuel_used": fuel_used,
        "low_power_days": low_time,
        "dv_mps": state.dv_mps,
    }
    return score, breakdown


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="IKAROS β-GO! (HTV style)", layout="wide")

# Reduce padding so HUD stays visible
st.markdown(
    """
<style>
.block-container {padding-top: 0.8rem; padding-bottom: 0.8rem;}
</style>
""",
    unsafe_allow_html=True
)

st.title("☀️ IKAROS β-GO!（HTV-GO風：目標軌道と予測）")
st.caption("SRP（太陽光輻射圧）は小さい。だから『大遷移』じゃなく『早めの微調整』で目標（ノミナル）に乗せる。")

# Sidebar: difficulty & teacher
with st.sidebar:
    st.header("設定")
    hard_mode = st.toggle("高難易度（通信中だけβ変更＋燃料制限）", value=False)
    show_teacher = st.toggle("先生モード（詳細）", value=False)
    st.divider()
    st.caption("※ SRPのみ（IESモード・事前角度指定は削除）")

cfg = Config(hard_mode=bool(hard_mode))

if show_teacher:
    with st.sidebar.expander("先生モード：パラメータ", expanded=False):
        cfg.dt_days = st.slider("刻み (days)", 0.2, 2.0, cfg.dt_days, 0.1)
        cfg.time_limit_days = st.slider("制限時間 (days)", 120.0, 420.0, cfg.time_limit_days, 5.0)
        cfg.a0 = st.slider("SRP強さ (AU/yr^2)", 0.005, 0.06, cfg.a0, 0.001)
        cfg.venus_phase_offset_deg = st.slider("金星位相ずれ (deg)", 0.0, 60.0, cfg.venus_phase_offset_deg, 1.0)
        cfg.comm_angle_deg = st.slider("通信に必要な地球角 (deg)", 5.0, 30.0, cfg.comm_angle_deg, 1.0)
        cfg.fuel_start = st.slider("燃料（姿勢リソース）", 40.0, 200.0, cfg.fuel_start, 5.0)
        cfg.fuel_per_deg = st.slider("燃料/deg", 0.05, 1.0, cfg.fuel_per_deg, 0.05)
        cfg.ticks_per_refresh = st.slider("自動進行の速さ", 1, 6, cfg.ticks_per_refresh, 1)

# One-time nominal precompute per config signature
cfg_sig = (cfg.dt_days, cfg.time_limit_days, cfg.a0, cfg.venus_phase_offset_deg)
if "nominal" not in st.session_state or st.session_state.get("cfg_sig") != cfg_sig:
    t_nom, r_nom, b_nom = simulate_nominal(cfg)
    st.session_state.nominal = dict(t_days=t_nom, r=r_nom, beta=b_nom)
    st.session_state.cfg_sig = cfg_sig

nominal = st.session_state.nominal

# Session init
if "sim" not in st.session_state:
    st.session_state.sim = init_state(cfg)

state: SimState = st.session_state.sim

def reset_all():
    st.session_state.sim = init_state(cfg)
    st.session_state.sim.phase = "play"
    st.session_state.sim.done = False
    st.session_state.sim.message = ""
    st.session_state.sim.score = 0.0

# Controls row
c1, c2, c3, c4, c5 = st.columns([1,1,1,1.2,1.2], gap="small")
with c1:
    if st.button("🔁 リセット", use_container_width=True):
        reset_all()
        state = st.session_state.sim

# Determine comm lock (for hard mode)
comm_ok = comm_window_active(state.x[:2], state.t_yr, cfg)
fuel_ok = state.fuel > 0.0

lock_reason = None
if cfg.hard_mode:
    if not comm_ok:
        lock_reason = "通信ウィンドウ外（高難易度）"
    elif not fuel_ok:
        lock_reason = "燃料切れ（高難易度）"

def apply_beta_change(new_beta: float):
    # Apply and consume fuel proportional to change
    new_beta = float(clamp(new_beta, -75.0, 75.0))
    d = abs(new_beta - state.beta_deg)
    cost = d * cfg.fuel_per_deg
    if cfg.hard_mode and lock_reason is not None:
        return  # locked
    if cfg.hard_mode and state.fuel - cost < 0:
        return
    state.beta_last_deg = state.beta_deg
    state.beta_deg = new_beta
    state.fuel = max(0.0, state.fuel - cost)

with c2:
    btn_left = st.button("⬅ β -5°", use_container_width=True, disabled=(cfg.hard_mode and lock_reason is not None))
    if btn_left:
        apply_beta_change(state.beta_deg - 5.0)
with c3:
    btn_right = st.button("➡ β +5°", use_container_width=True, disabled=(cfg.hard_mode and lock_reason is not None))
    if btn_right:
        apply_beta_change(state.beta_deg + 5.0)
with c4:
    preset = st.selectbox("ワンタッチβ", ["-35°（内側へ）", "-20°（ノミナル寄り）", "0°（待ち）", "+20°（外側へ）"], index=1)
    if preset.startswith("-35"):
        apply_beta_change(-35.0)
    elif preset.startswith("-20"):
        apply_beta_change(-20.0)
    elif preset.startswith("0"):
        apply_beta_change(0.0)
    else:
        apply_beta_change(20.0)
with c5:
    # Autoplay toggle + manual step
    colA, colB = st.columns(2)
    with colA:
        if st.button("⏸/▶ 自動進行", use_container_width=True):
            st.session_state.auto = not st.session_state.get("auto", True)
    with colB:
        if st.button("▶ 進める（約5日）", use_container_width=True):
            st.session_state.manual_step = True

auto = st.session_state.get("auto", True)
manual_step = st.session_state.get("manual_step", False)
st.session_state.manual_step = False

# Always-visible HUD (β always visible)
t_days = state.t_yr * 365.25
r_sc = state.x[:2]
d_venus = venus_distance(r_sc, state.t_yr, cfg)
pwr = power_frac(r_sc, state.beta_deg)
ea = earth_angle_deg(r_sc, state.t_yr)

# nominal reference at current time (nearest index)
idx = int(round(t_days / cfg.dt_days))
idx = int(clamp(idx, 0, len(nominal["t_days"]) - 1))
r_nom_now = nominal["r"][idx]

orbit_err = norm(r_sc - r_nom_now)

m1, m2, m3, m4, m5, m6 = st.columns([1,1,1,1,1,1], gap="small")
m1.metric("β角", f"{state.beta_deg:+.0f}°")
m2.metric("発電量", f"{pwr:.2f}", delta=("OK" if pwr >= 0.75 else "不足"))
m3.metric("地球角", f"{ea:.1f}°", delta=("COMMS" if comm_ok else "NO-COMMS"))
m4.metric("軌道誤差", f"{orbit_err:.3f} AU")
m5.metric("燃料", f"{state.fuel:.1f}")
m6.metric("金星距離", f"{d_venus:.3f} AU", delta=f"ベスト {state.best_dist:.3f}")

if cfg.hard_mode and lock_reason is not None:
    st.warning(f"β変更ロック中：{lock_reason}")

# Main layout: plot + live charts
left, right = st.columns([1.35, 1.0], gap="large")

# --- Simulation stepping function
def integrate_steps(n_steps: int):
    dt_yr = cfg.dt_days / 365.25

    for _ in range(n_steps):
        # stop at time limit
        if state.t_yr * 365.25 >= cfg.time_limit_days:
            state.done = True
            state.phase = "result"
            state.message = "シミュレーション終了（制限時間）"
            state.score, breakdown = compute_score(state, cfg)
            st.session_state.breakdown = breakdown
            break

        # integrate
        a_prop = srp_accel(state.x[:2], state.beta_deg, cfg.a0)
        a_mps2 = norm(a_prop) * AUYR2_TO_MPS2
        dt_s = dt_yr * YR_S
        state.dv_mps += a_mps2 * dt_s

        state.x = rk4_step(state.x, beta_deg=state.beta_deg, a0=cfg.a0, dt=dt_yr)
        state.t_yr += dt_yr

        # logs
        t_days_now = state.t_yr * 365.25
        idx2 = int(round(t_days_now / cfg.dt_days))
        idx2 = int(clamp(idx2, 0, len(nominal["t_days"]) - 1))
        r_nom = nominal["r"][idx2]
        update_logs(state, cfg, r_nom)


# Initial log at start
if len(state.t_days_log) == 0 and state.phase == "play":
    update_logs(state, cfg, r_nom_now)

# Auto advance (HTV-GO feel)
if state.phase == "play" and (not state.done):
    if auto and HAVE_AUTOREFRESH:
        st_autorefresh(interval=350, key="tick")
        integrate_steps(cfg.ticks_per_refresh)
    elif manual_step:
        integrate_steps(int(max(1, round(5.0 / cfg.dt_days))))

# --- Plot: nominal / actual / prediction
with left:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (AU)")
    ax.set_ylabel("y (AU)")

    th = np.linspace(0, 2*math.pi, 500)
    ax.plot(R_E*np.cos(th), R_E*np.sin(th), linewidth=1)
    ax.plot(R_V*np.cos(th), R_V*np.sin(th), linewidth=1)

    # nominal trajectory (dashed)
    rN = nominal["r"]
    ax.plot(rN[:,0], rN[:,1], linestyle="--", linewidth=1.5)

    # actual trajectory
    r_log = np.array(state.r_log, dtype=float) if state.r_log else np.zeros((0,2))
    if len(r_log) >= 2:
        ax.plot(r_log[:,0], r_log[:,1], linewidth=2)

    # prediction (dotted)
    pred = predict_future(state, cfg, horizon_days=70.0)
    ax.plot(pred[:,0], pred[:,1], linestyle=":", linewidth=2)

    # bodies
    ax.scatter([0.0], [0.0], marker="o")
    ax.text(0.02, 0.02, "Sun", fontsize=10)

    r_e = planet_pos(R_E, N_E, state.t_yr, 0.0)
    ax.scatter([r_e[0]], [r_e[1]], marker="o")
    ax.text(r_e[0]+0.02, r_e[1]+0.02, "Earth", fontsize=10)

    r_v = planet_pos(R_V, N_V, state.t_yr, venus_phase(cfg))
    ax.scatter([r_v[0]], [r_v[1]], marker="o")
    ax.text(r_v[0]+0.02, r_v[1]+0.02, "Venus", fontsize=10)

    r_sc_now = state.x[:2]
    ax.scatter([r_sc_now[0]], [r_sc_now[1]], marker="x")
    ax.text(r_sc_now[0]+0.02, r_sc_now[1]+0.02, f"Sail β={state.beta_deg:+.0f}°", fontsize=10)

    # acceleration arrow (tiny, but visible)
    a_vec = srp_accel(r_sc_now, state.beta_deg, cfg.a0)
    if norm(a_vec) > 1e-12:
        a_hat = unit(a_vec)
        ax.arrow(r_sc_now[0], r_sc_now[1], 0.11*a_hat[0], 0.11*a_hat[1], head_width=0.03, length_includes_head=True)

    lim = 1.35
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    ax.set_title("実線=自機 / 破線=ノミナル / 点線=このまま(β固定)の予測")
    st.pyplot(fig, use_container_width=True)

# --- Right panel: live time series + tips
with right:
    st.subheader("ライブ表示")
    st.caption("（HTV-GO風：数字＋時系列で『今の操作が効いてるか』を見る）")

    if len(state.t_days_log) >= 2:
        t = np.array(state.t_days_log, dtype=float)

        # 2x2 mini charts
        g1, g2 = st.columns(2)
        with g1:
            st.line_chart({"発電": np.array(state.power_log, dtype=float)}, height=160)
        with g2:
            st.line_chart({"地球角(deg)": np.array(state.earth_angle_log, dtype=float)}, height=160)

        g3, g4 = st.columns(2)
        with g3:
            st.line_chart({"軌道誤差(AU)": np.array(state.orbit_err_log, dtype=float)}, height=160)
        with g4:
            st.line_chart({"燃料": np.array(state.fuel_log, dtype=float)}, height=160)
    else:
        st.info("時間を進めるとグラフが出ます。")

    st.divider()
    st.subheader("コツ")
    st.markdown(
        """
- **ノミナル（破線）**に乗せる気持ちでβを微調整  
- **点線（予測）**がノミナルから離れるなら、βを変える  
- SRPは小さい：**早い修正が勝つ**（遅れるほど追いつかない）
"""
    )
    if cfg.hard_mode:
        st.warning("高難易度：通信ウィンドウ（地球角がしきい値以上）でしかβを変えられません。")

# -----------------------------
# Result screen
# -----------------------------
if state.phase == "result":
    st.divider()
    st.header("📊 リザルト")

    breakdown = st.session_state.get("breakdown", {})
    st.subheader(f"スコア：{state.score:.0f} 点")
    st.write(f"- 最小 金星距離：{breakdown.get('best_dist_AU', state.best_dist):.3f} AU")
    st.write(f"- ΔV相当：{breakdown.get('dv_mps', state.dv_mps):.0f} m/s（SRPの小ささを体感）")
    st.write(f"- 燃料使用：{breakdown.get('fuel_used', cfg.fuel_start - state.fuel):.1f}")
    st.write(f"- 低発電時間：{breakdown.get('low_power_days', 0.0):.1f} days")
    st.write("")
    st.caption("スコア内訳（目安）：接近ボーナス + ΔVボーナス − 軌道誤差 − 低発電 − 燃料")

    # Detailed plots
    t = np.array(state.t_days_log, dtype=float) if state.t_days_log else np.array([0.0])

    # Build a simple results dashboard with multiple plots
    def plot_series(y: List[float], ylabel: str, title: str):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel("day")
        ax.set_ylabel(ylabel)
        ax.plot(t, np.array(y, dtype=float))
        ax.set_title(title)
        st.pyplot(fig, use_container_width=True)

    rcol1, rcol2 = st.columns(2, gap="large")
    with rcol1:
        plot_series(state.power_log, "power (rel)", "発電量（相対）")
        plot_series(state.orbit_err_log, "error (AU)", "軌道誤差（ノミナルとの差）")
        plot_series(state.dist_venus_log, "dist (AU)", "金星まで距離")
    with rcol2:
        plot_series(state.earth_angle_log, "deg", "地球角（通信のしやすさ）")
        plot_series(state.fuel_log, "fuel", "燃料残量（姿勢リソース）")
        plot_series(state.beta_log, "deg", "β角の操作履歴")

    st.divider()
    if st.button("もう一回やる（リセット）", use_container_width=True):
        reset_all()
        (st.rerun() if hasattr(st, 'rerun') else st.experimental_rerun())

# IKAROS β-GO! (HTV-GO style) - SRP only (v3)
# ------------------------------------------------------------
# Fixes / improvements:
# - Matplotlib Japanese: use japanize_matplotlib.
# - Live charts: add clear labels.
# - Throttle refresh & lighter rendering:
#   - default refresh slower
#   - orbit map is optional (toggle) because it's heavy
#   - downsample plotting points
# - Main view emphasizes "orbit error vs time" with nominal (0) + prediction overlay.
#
# Features:
# - Nominal (target) trajectory shown (dashed) in optional orbit map.
# - Prediction "if keep current β" shown (dotted) in optional orbit map.
# - Live: power, Earth angle, orbit error time-series, fuel, ΔV.
# - After time limit: result screen with graphs + score.
# - SRP only (no IES, no pre-plan mode).
# - Optional high difficulty: β changes only during comm window, fuel limited.
# ------------------------------------------------------------

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

# Japanese font support for matplotlib
try:
    import japanize_matplotlib  # noqa: F401
except Exception:
    # If it fails, we will avoid Japanese inside matplotlib as much as possible.
    pass

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

def downsample_xy(xy: np.ndarray, max_points: int = 1200) -> np.ndarray:
    if xy.shape[0] <= max_points:
        return xy
    stride = int(math.ceil(xy.shape[0] / max_points))
    return xy[::stride]


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

AU_M = 1.495978707e11
YR_S = 365.25 * 24 * 3600.0
AUYR2_TO_MPS2 = AU_M / (YR_S**2)


def planet_pos(r: float, n: float, t_yr: float, phase: float) -> np.ndarray:
    a = n * t_yr + phase
    return np.array([r * math.cos(a), r * math.sin(a)], dtype=float)

def planet_vel(r: float, n: float, t_yr: float, phase: float) -> np.ndarray:
    a = n * t_yr + phase
    return np.array([-r * n * math.sin(a), r * n * math.cos(a)], dtype=float)


# -----------------------------
# Baseline transfer-like start (Hohmann-ish)
# -----------------------------
A_H = 0.5 * (R_E + R_V)
T_HALF_YR = 0.5 * math.sqrt(A_H**3)  # because MU=4π²
T_HALF_DAYS = T_HALF_YR * 365.25

def hohmann_aphelion_speed() -> float:
    # v = sqrt(mu*(2/r - 1/a))
    r = R_E
    a = A_H
    return math.sqrt(MU * (2.0 / r - 1.0 / a))

V_APH = hohmann_aphelion_speed()

def aligned_venus_phase() -> float:
    return (math.pi - N_V * T_HALF_YR) % (2 * math.pi)


# -----------------------------
# SRP model (β only, in-plane)
# -----------------------------
def power_frac(r_sc: np.ndarray, beta_deg: float) -> float:
    # power ∝ (1/r^2) * cos(|β|)
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
# Nominal plan -> nominal trajectory
# -----------------------------
def nominal_beta_schedule(t_days: float) -> float:
    # A simple fixed plan used only for teaching "nominal vs actual".
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
    time_limit_days: float = 220.0
    a0: float = 0.020  # AU/yr^2 at 1AU beta=0
    venus_phase_offset_deg: float = 18.0
    comm_angle_deg: float = 12.0
    fuel_start: float = 120.0
    fuel_per_deg: float = 0.35  # cost per degree change
    hard_mode: bool = False
    refresh_ms: int = 900
    ticks_per_refresh: int = 1
    show_orbit_map: bool = False

@dataclass
class SimState:
    t_yr: float
    x: np.ndarray  # [x,y,vx,vy]
    beta_deg: float
    fuel: float
    dv_mps: float
    phase: str  # "play" or "result"
    score: float
    # logs (sampled every integration step)
    r_log: List[np.ndarray]
    t_days_log: List[float]
    beta_log: List[float]
    power_log: List[float]
    earth_angle_log: List[float]
    orbit_err_log: List[float]
    fuel_log: List[float]
    dv_log: List[float]
    dist_venus_log: List[float]
    best_dist: float

def init_state(cfg: Config) -> SimState:
    r0 = np.array([R_E, 0.0], dtype=float)
    v0 = np.array([0.0, V_APH], dtype=float)
    x0 = np.array([r0[0], r0[1], v0[0], v0[1]], dtype=float)

    return SimState(
        t_yr=0.0,
        x=x0,
        beta_deg=-20.0,
        fuel=cfg.fuel_start,
        dv_mps=0.0,
        phase="play",
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
# Derived quantities
# -----------------------------
def venus_phase(cfg: Config) -> float:
    return aligned_venus_phase() + rad(cfg.venus_phase_offset_deg)

def venus_distance(r_sc: np.ndarray, t_yr: float, cfg: Config) -> float:
    r_v = planet_pos(R_V, N_V, t_yr, venus_phase(cfg))
    return norm(r_sc - r_v)

def earth_angle_deg(r_sc: np.ndarray, t_yr: float) -> float:
    r_e = planet_pos(R_E, N_E, t_yr, 0.0)
    to_sun = -r_sc
    to_earth = r_e - r_sc
    return deg(angle_between(to_sun, to_earth))

def comm_window_active(r_sc: np.ndarray, t_yr: float, cfg: Config) -> bool:
    return earth_angle_deg(r_sc, t_yr) >= cfg.comm_angle_deg


# -----------------------------
# Nominal trajectory precompute
# -----------------------------
@st.cache_data(show_spinner=False)
def simulate_nominal(dt_days: float, time_limit_days: float, a0: float, venus_phase_offset_deg: float) -> Dict[str, np.ndarray]:
    cfg = Config(dt_days=dt_days, time_limit_days=time_limit_days, a0=a0, venus_phase_offset_deg=venus_phase_offset_deg)
    dt_yr = cfg.dt_days / 365.25
    steps = int(math.ceil(cfg.time_limit_days / cfg.dt_days)) + 1

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

    return {"t_days": t_days, "r": r_nom, "beta": b_nom}


def nominal_r_at(t_days: float, nominal: Dict[str, np.ndarray], dt_days: float) -> np.ndarray:
    idx = int(round(t_days / dt_days))
    idx = int(clamp(idx, 0, len(nominal["t_days"]) - 1))
    return nominal["r"][idx]


def predict_error_timeseries(state: SimState, cfg: Config, nominal: Dict[str, np.ndarray], horizon_days: float = 70.0) -> pd.DataFrame:
    """Return dataframe with columns: day, err_now (NaN for future), err_pred."""
    dt_yr = cfg.dt_days / 365.25
    steps = int(max(2, round(horizon_days / cfg.dt_days)))

    x = state.x.copy()
    t_yr = state.t_yr

    days = []
    err_pred = []

    for _ in range(steps):
        td = t_yr * 365.25
        r_nom = nominal_r_at(td, nominal, cfg.dt_days)
        err_pred.append(norm(x[:2] - r_nom))
        days.append(td)
        x = rk4_step(x, beta_deg=state.beta_deg, a0=cfg.a0, dt=dt_yr)
        t_yr += dt_yr

    return pd.DataFrame({"day": np.array(days), "予測誤差(AU)": np.array(err_pred)})


# -----------------------------
# Scoring
# -----------------------------
def compute_score(state: SimState, cfg: Config) -> Tuple[float, Dict[str, float]]:
    best = state.best_dist
    approach = 500.0 * math.exp(-best / 0.08)

    if len(state.orbit_err_log) > 2:
        err_int = float(np.trapz(np.array(state.orbit_err_log), x=np.array(state.t_days_log)))
    else:
        err_int = 0.0
    track_pen = 1200.0 * err_int

    t = np.array(state.t_days_log) if state.t_days_log else np.array([0.0])
    p = np.array(state.power_log) if state.power_log else np.array([1.0])
    low = (p < 0.75).astype(float)
    low_time = float(np.trapz(low, x=t)) if len(t) > 2 else 0.0
    power_pen = 25.0 * low_time

    fuel_used = cfg.fuel_start - state.fuel
    fuel_pen = 2.2 * max(0.0, fuel_used)

    dv_bonus = min(35.0, state.dv_mps * 0.08)

    score = max(0.0, approach + dv_bonus - track_pen - power_pen - fuel_pen)

    return score, {
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


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="IKAROS β-GO! (HTV style)", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 0.7rem; padding-bottom: 0.7rem;}
</style>
""",
    unsafe_allow_html=True
)

st.title("☀️ IKAROS β-GO!（HTV-GO風：ノミナル＆予測つき）")
st.caption("SRP（太陽光輻射圧）は小さい。だから『大遷移』じゃなく『早めの微調整』でノミナルに乗せる。")

with st.sidebar:
    st.header("設定")
    hard_mode = st.toggle("高難易度（通信中だけβ変更＋燃料制限）", value=False)
    show_orbit_map = st.toggle("軌道図を表示（重い）", value=False)
    st.divider()
    st.subheader("更新頻度（軽量化）")
    refresh_ms = st.slider("更新間隔 (ms)", 400, 2000, 900, 50)
    ticks_per_refresh = st.slider("1回の更新で進める回数", 1, 6, 1, 1)
    st.divider()
    show_teacher = st.toggle("先生モード（詳細）", value=False)
    st.caption("※ SRPのみ（IES／事前角度指定は無し）")

cfg = Config(
    hard_mode=bool(hard_mode),
    refresh_ms=int(refresh_ms),
    ticks_per_refresh=int(ticks_per_refresh),
    show_orbit_map=bool(show_orbit_map),
)

if show_teacher:
    with st.sidebar.expander("先生モード：パラメータ", expanded=False):
        cfg.dt_days = st.slider("刻み (days)", 0.2, 2.0, cfg.dt_days, 0.1)
        cfg.time_limit_days = st.slider("制限時間 (days)", 120.0, 420.0, cfg.time_limit_days, 5.0)
        cfg.a0 = st.slider("SRP強さ (AU/yr^2)", 0.005, 0.06, cfg.a0, 0.001)
        cfg.venus_phase_offset_deg = st.slider("金星位相ずれ (deg)", 0.0, 60.0, cfg.venus_phase_offset_deg, 1.0)
        cfg.comm_angle_deg = st.slider("通信に必要な地球角 (deg)", 5.0, 30.0, cfg.comm_angle_deg, 1.0)
        cfg.fuel_start = st.slider("燃料（姿勢リソース）", 40.0, 200.0, cfg.fuel_start, 5.0)
        cfg.fuel_per_deg = st.slider("燃料/deg", 0.05, 1.0, cfg.fuel_per_deg, 0.05)

# Precompute nominal
nominal = simulate_nominal(cfg.dt_days, cfg.time_limit_days, cfg.a0, cfg.venus_phase_offset_deg)

# Session init
if "sim" not in st.session_state:
    st.session_state.sim = init_state(cfg)
    st.session_state.auto = True

state: SimState = st.session_state.sim

def reset_all():
    st.session_state.sim = init_state(cfg)
    st.session_state.auto = True

# Controls
c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1.2, 1.2], gap="small")
with c1:
    if st.button("🔁 リセット", use_container_width=True):
        reset_all()
        state = st.session_state.sim

comm_ok = comm_window_active(state.x[:2], state.t_yr, cfg)
lock = cfg.hard_mode and (not comm_ok or state.fuel <= 0.0)
if cfg.hard_mode and not comm_ok:
    lock_reason = "通信ウィンドウ外"
elif cfg.hard_mode and state.fuel <= 0.0:
    lock_reason = "燃料切れ"
else:
    lock_reason = ""

def apply_beta(new_beta: float):
    if lock:
        return
    new_beta = float(clamp(new_beta, -75.0, 75.0))
    d = abs(new_beta - state.beta_deg)
    cost = d * cfg.fuel_per_deg
    if cfg.hard_mode and state.fuel - cost < 0:
        return
    state.beta_deg = new_beta
    state.fuel = max(0.0, state.fuel - cost)

with c2:
    if st.button("⬅ β -5°", use_container_width=True, disabled=lock):
        apply_beta(state.beta_deg - 5.0)
with c3:
    if st.button("➡ β +5°", use_container_width=True, disabled=lock):
        apply_beta(state.beta_deg + 5.0)
with c4:
    preset = st.selectbox("ワンタッチβ", ["-35°（内側へ）", "-20°（ノミナル寄り）", "0°（待ち）", "+20°（外側へ）"], index=1)
    if preset.startswith("-35"):
        apply_beta(-35.0)
    elif preset.startswith("-20"):
        apply_beta(-20.0)
    elif preset.startswith("0"):
        apply_beta(0.0)
    else:
        apply_beta(20.0)
with c5:
    colA, colB = st.columns(2)
    with colA:
        if st.button("⏸/▶ 自動進行", use_container_width=True):
            st.session_state.auto = not st.session_state.auto
    with colB:
        manual = st.button("▶ 進める（約5日）", use_container_width=True)

if cfg.hard_mode and lock_reason:
    st.warning(f"β変更ロック中（高難易度）：{lock_reason}")

# HUD (always visible)
t_days = state.t_yr * 365.25
r_sc = state.x[:2]
pwr = power_frac(r_sc, state.beta_deg)
ea = earth_angle_deg(r_sc, state.t_yr)
d_venus = venus_distance(r_sc, state.t_yr, cfg)
r_nom_now = nominal_r_at(t_days, nominal, cfg.dt_days)
orbit_err = norm(r_sc - r_nom_now)

m1, m2, m3, m4, m5, m6 = st.columns([1, 1, 1, 1, 1, 1], gap="small")
m1.metric("β角", f"{state.beta_deg:+.0f}°")
m2.metric("発電量", f"{pwr:.2f}", delta=("OK" if pwr >= 0.75 else "不足"))
m3.metric("地球角", f"{ea:.1f}°", delta=("COMMS" if comm_ok else "NO-COMMS"))
m4.metric("軌道誤差", f"{orbit_err:.3f} AU")
m5.metric("燃料", f"{state.fuel:.1f}")
m6.metric("金星距離", f"{d_venus:.3f} AU", delta=f"ベスト {state.best_dist:.3f}")

# -----------------------------
# Simulation stepping
# -----------------------------
def log_step():
    r_sc2 = state.x[:2]
    td = state.t_yr * 365.25
    rn = nominal_r_at(td, nominal, cfg.dt_days)

    state.r_log.append(r_sc2.copy())
    state.t_days_log.append(float(td))
    state.beta_log.append(float(state.beta_deg))
    state.power_log.append(float(power_frac(r_sc2, state.beta_deg)))
    state.earth_angle_log.append(float(earth_angle_deg(r_sc2, state.t_yr)))
    state.orbit_err_log.append(float(norm(r_sc2 - rn)))
    state.fuel_log.append(float(state.fuel))
    state.dv_log.append(float(state.dv_mps))

    dv = venus_distance(r_sc2, state.t_yr, cfg)
    state.dist_venus_log.append(float(dv))
    state.best_dist = min(state.best_dist, dv)

def integrate_steps(n_steps: int):
    dt_yr = cfg.dt_days / 365.25
    for _ in range(n_steps):
        if state.t_yr * 365.25 >= cfg.time_limit_days:
            state.phase = "result"
            state.score, breakdown = compute_score(state, cfg)
            st.session_state.breakdown = breakdown
            break

        a_prop = srp_accel(state.x[:2], state.beta_deg, cfg.a0)
        a_mps2 = norm(a_prop) * AUYR2_TO_MPS2
        state.dv_mps += a_mps2 * (dt_yr * YR_S)

        state.x = rk4_step(state.x, beta_deg=state.beta_deg, a0=cfg.a0, dt=dt_yr)
        state.t_yr += dt_yr

        log_step()

# initial log
if state.phase == "play" and len(state.t_days_log) == 0:
    log_step()

# Autoplay
auto = st.session_state.get("auto", True)
if state.phase == "play":
    if auto and HAVE_AUTOREFRESH:
        st_autorefresh(interval=cfg.refresh_ms, key="tick")
        integrate_steps(cfg.ticks_per_refresh)
    elif manual:
        integrate_steps(int(max(1, round(5.0 / cfg.dt_days))))

# -----------------------------
# Main view (HTV-GO-like): orbit error big + (optional) orbit map
# -----------------------------
left, right = st.columns([1.35, 1.0], gap="large")

with left:
    st.subheader("軌道誤差（大事）")
    st.caption("ノミナルとの差。0に近いほど『予定通り』。点線は“このままβ固定ならどうなる？”予測。")

    # Actual error series
    if len(state.t_days_log) >= 2:
        df_err = pd.DataFrame({
            "day": np.array(state.t_days_log, dtype=float),
            "誤差(AU)": np.array(state.orbit_err_log, dtype=float),
        })

        df_pred = predict_error_timeseries(state, cfg, nominal, horizon_days=70.0)

        # Merge for a single chart (Vega) with clear labels
        # Create a wide df with index day, then plot
        df = df_err.set_index("day")
        df_pred2 = df_pred.set_index("day")
        # Align index union
        df_all = df.join(df_pred2, how="outer")
        df_all["ノミナル(0)"] = 0.0

        st.line_chart(df_all, height=340, use_container_width=True)

        if cfg.show_orbit_map:
            st.divider()
            st.subheader("軌道図（重い）")
            st.caption("実線=自機 / 破線=ノミナル / 点線=予測（β固定）")

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x (AU)")
            ax.set_ylabel("y (AU)")

            th = np.linspace(0, 2*math.pi, 500)
            ax.plot(R_E*np.cos(th), R_E*np.sin(th), linewidth=1)
            ax.plot(R_V*np.cos(th), R_V*np.sin(th), linewidth=1)

            # nominal dashed (downsample)
            rN = downsample_xy(nominal["r"], 1200)
            ax.plot(rN[:,0], rN[:,1], linestyle="--", linewidth=1.5)

            # actual
            rA = downsample_xy(np.array(state.r_log, dtype=float), 1200)
            if rA.shape[0] >= 2:
                ax.plot(rA[:,0], rA[:,1], linewidth=2)

            # prediction dotted (downsample)
            # (reuse prediction simulation, but output positions)
            pred_pos = []
            dt_yr = cfg.dt_days / 365.25
            x = state.x.copy()
            for _ in range(int(max(2, round(70.0 / cfg.dt_days)))):
                pred_pos.append(x[:2].copy())
                x = rk4_step(x, beta_deg=state.beta_deg, a0=cfg.a0, dt=dt_yr)
            pred_pos = downsample_xy(np.array(pred_pos, dtype=float), 700)
            ax.plot(pred_pos[:,0], pred_pos[:,1], linestyle=":", linewidth=2)

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

            # acceleration arrow
            a_vec = srp_accel(r_sc_now, state.beta_deg, cfg.a0)
            if norm(a_vec) > 1e-12:
                a_hat = unit(a_vec)
                ax.arrow(r_sc_now[0], r_sc_now[1], 0.11*a_hat[0], 0.11*a_hat[1], head_width=0.03, length_includes_head=True)

            lim = 1.35
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)

            st.pyplot(fig, use_container_width=True)
    else:
        st.info("まだデータが少ないです。自動進行または『進める』で動かしてください。")

with right:
    st.subheader("ライブ表示（何のグラフ？）")
    if len(state.t_days_log) >= 2:
        # Show as 2x2 with clear captions
        g1, g2 = st.columns(2, gap="small")
        with g1:
            st.caption("発電量（相対）")
            st.line_chart(pd.DataFrame({"発電量": state.power_log}, index=state.t_days_log), height=160)
        with g2:
            st.caption("地球角（通信しやすさ）")
            st.line_chart(pd.DataFrame({"地球角[deg]": state.earth_angle_log}, index=state.t_days_log), height=160)

        g3, g4 = st.columns(2, gap="small")
        with g3:
            st.caption("軌道誤差（ノミナルとの差）")
            st.line_chart(pd.DataFrame({"誤差[AU]": state.orbit_err_log}, index=state.t_days_log), height=160)
        with g4:
            st.caption("燃料（姿勢リソース）")
            st.line_chart(pd.DataFrame({"燃料": state.fuel_log}, index=state.t_days_log), height=160)

        st.divider()
        st.caption("SRPの“効きの小ささ”を数字で見る")
        st.line_chart(pd.DataFrame({"ΔV相当[m/s]": state.dv_log}, index=state.t_days_log), height=170)
    else:
        st.info("時間を進めるとグラフが出ます。")

    st.divider()
    st.subheader("コツ")
    st.markdown(
        """
- **誤差グラフ**が上がり続けるなら、βを変えて“予測”を良くする  
- SRPは小さい：**早い修正が勝つ**（遅れるほど追いつかない）  
- 高難易度は、**通信できる時だけ**βが触れるので“先読み”が必要
"""
    )

# -----------------------------
# Result screen
# -----------------------------
if state.phase == "result":
    st.divider()
    st.header("📊 リザルト")

    breakdown = st.session_state.get("breakdown", {})
    st.subheader(f"スコア：{state.score:.0f} 点")
    st.write(f"- 最小 金星距離：{breakdown.get('best_dist_AU', state.best_dist):.3f} AU")
    st.write(f"- ΔV相当：{breakdown.get('dv_mps', state.dv_mps):.0f} m/s（SRPの小ささ）")
    st.write(f"- 燃料使用：{breakdown.get('fuel_used', cfg.fuel_start - state.fuel):.1f}")
    st.write(f"- 低発電時間：{breakdown.get('low_power_days', 0.0):.1f} days")
    st.caption("スコア目安：接近 + ΔV −（誤差積分 + 低発電 + 燃料）")

    # Use matplotlib for detailed plots (Japanese ok via japanize_matplotlib)
    t = np.array(state.t_days_log, dtype=float) if state.t_days_log else np.array([0.0])

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
        plot_series(state.earth_angle_log, "deg", "地球角（通信しやすさ）")
        plot_series(state.fuel_log, "fuel", "燃料（姿勢リソース）")
        plot_series(state.beta_log, "deg", "β角の履歴")

    st.divider()
    if st.button("もう一回やる（リセット）", use_container_width=True):
        reset_all()
        # rerun safely
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()

st.caption("※ 教材用の簡略モデルです（実機の精密な航法・姿勢・光圧モデルではありません）。")

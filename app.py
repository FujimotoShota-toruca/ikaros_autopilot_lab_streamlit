# IKAROS β-GO! (HTV-GO style, ALL Vega/Altair) - SRP only
# ------------------------------------------------------------
# v5 fixes & upgrades:
# - FIX: numpy 2.x removed np.trapz in some builds -> use our own trapezoid integrator.
# - Add Orbit (2D) chart (Altair): actual / nominal / prediction + Earth/Venus orbits.
# - Reduce legend duplication in main error chart.
# - Add initial offset (position/velocity) for better game balance: "do nothing -> bad".
# - Keep everything Vega/Altair (no Matplotlib).
# ------------------------------------------------------------

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import streamlit as st
import altair as alt

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

def trapz_integral(y: List[float], x: List[float]) -> float:
    """Trapezoidal integration without relying on numpy.trapz/trapezoid."""
    if len(y) < 2 or len(x) < 2:
        return float(y[-1]) if y else 0.0
    s = 0.0
    for i in range(1, len(y)):
        dx = float(x[i] - x[i-1])
        s += 0.5 * dx * float(y[i] + y[i-1])
    return s

def downsample_indices(n: int, max_n: int) -> List[int]:
    if n <= max_n:
        return list(range(n))
    step = int(math.ceil(n / max_n))
    return list(range(0, n, step))


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

# Unit conversions (for SRP smallness)
AU_M = 1.495978707e11
YR_S = 365.25 * 24 * 3600.0
AUYR2_TO_MPS2 = AU_M / (YR_S**2)

def planet_pos(r: float, n: float, t_yr: float, phase: float) -> np.ndarray:
    a = n * t_yr + phase
    return np.array([r * math.cos(a), r * math.sin(a)], dtype=float)


# -----------------------------
# Baseline transfer-like start (Hohmann-ish)
# -----------------------------
A_H = 0.5 * (R_E + R_V)
T_HALF_YR = 0.5 * math.sqrt(A_H**3)  # because MU=4π²
T_HALF_DAYS = T_HALF_YR * 365.25

def hohmann_aphelion_speed() -> float:
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
    return float(np.clip(math.cos(abs(rad(beta_deg))) / (r*r), 0.0, 2.0))

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
# Nominal plan (simple schedule)
# -----------------------------
def nominal_beta_schedule(t_days: float) -> float:
    # Change points create "guidance phases" (you can tweak later)
    if t_days < 70:
        return -20.0
    if t_days < 100:
        return 0.0
    return -10.0


# -----------------------------
# Config / State
# -----------------------------
@dataclass
class Config:
    # simulation
    dt_days: float = 0.5
    time_limit_days: float = 220.0
    pred_horizon_days: float = 90.0
    nominal_padding_days: float = 180.0  # must exceed pred horizon to avoid weird reference
    a0: float = 0.020

    # mission-ish constraints
    comm_angle_deg: float = 12.0
    power_ok: float = 0.75
    err_ok: float = 0.008  # tighter by default

    # ops resource (attitude/ops)
    fuel_start: float = 120.0
    fuel_per_deg: float = 0.35
    hard_mode: bool = False

    # refresh
    refresh_ms: int = 1000
    ticks_per_refresh: int = 1

    # initial offset (game balance)
    init_pos_offset_au: float = 0.006   # position error magnitude
    init_vel_offset_auyr: float = 0.000 # velocity error magnitude
    init_beta_deg: float = 0.0          # start β away from nominal

@dataclass
class SimState:
    t_yr: float
    x: np.ndarray
    beta_deg: float
    fuel: float
    dv_mps: float
    accel_ums2: float
    phase: str  # play/result

    # logs
    t_days: List[float]
    rx: List[float]
    ry: List[float]
    beta: List[float]
    power: List[float]
    earth_angle: List[float]
    orbit_err: List[float]
    fuel_log: List[float]
    dv_log: List[float]
    accel_log_ums2: List[float]


def earth_angle_deg(r_sc: np.ndarray, t_yr: float) -> float:
    r_e = planet_pos(R_E, N_E, t_yr, 0.0)
    to_sun = -r_sc
    to_earth = r_e - r_sc
    return deg(angle_between(to_sun, to_earth))

def comm_window_active(r_sc: np.ndarray, t_yr: float, cfg: Config) -> bool:
    return earth_angle_deg(r_sc, t_yr) >= cfg.comm_angle_deg


def init_state(cfg: Config) -> SimState:
    # Nominal initial heliocentric state
    r0 = np.array([R_E, 0.0], dtype=float)
    v0 = np.array([0.0, V_APH], dtype=float)

    # Add a deliberate navigation/dispersion offset (game balance)
    # (simple: +y position offset; +x velocity offset)
    r0 = r0 + np.array([0.0, cfg.init_pos_offset_au], dtype=float)
    v0 = v0 + np.array([cfg.init_vel_offset_auyr, 0.0], dtype=float)

    x0 = np.array([r0[0], r0[1], v0[0], v0[1]], dtype=float)

    return SimState(
        t_yr=0.0,
        x=x0,
        beta_deg=float(cfg.init_beta_deg),
        fuel=float(cfg.fuel_start),
        dv_mps=0.0,
        accel_ums2=0.0,
        phase="play",
        t_days=[],
        rx=[],
        ry=[],
        beta=[],
        power=[],
        earth_angle=[],
        orbit_err=[],
        fuel_log=[],
        dv_log=[],
        accel_log_ums2=[],
    )


# -----------------------------
# Nominal precompute (cached)
# -----------------------------
@st.cache_data(show_spinner=False)
def simulate_nominal_cached(dt_days: float, time_limit_days: float, padding_days: float, a0: float) -> Dict[str, np.ndarray]:
    dt_yr = dt_days / 365.25
    total_days = time_limit_days + padding_days
    steps = int(math.ceil(total_days / dt_days)) + 1

    r0 = np.array([R_E, 0.0], dtype=float)
    v0 = np.array([0.0, V_APH], dtype=float)
    x = np.array([r0[0], r0[1], v0[0], v0[1]], dtype=float)

    t_days = np.zeros(steps, dtype=float)
    r_nom = np.zeros((steps, 2), dtype=float)

    t_yr = 0.0
    for i in range(steps):
        td = t_yr * 365.25
        b = nominal_beta_schedule(td)
        t_days[i] = td
        r_nom[i] = x[:2]
        x = rk4_step(x, beta_deg=b, a0=a0, dt=dt_yr)
        t_yr += dt_yr

    return {"t_days": t_days, "r": r_nom}

def nominal_at_time(nom: Dict[str, np.ndarray], cfg: Config, t_days: float) -> np.ndarray:
    idx = int(round(t_days / cfg.dt_days))
    idx = int(clamp(idx, 0, len(nom["t_days"]) - 1))
    return nom["r"][idx]


def predict_future(state: SimState, cfg: Config, nom: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (t_future_days, err_future, r_future[steps,2]) under constant β from current state."""
    # Safety: ensure nominal padding is enough even if teacher changes
    pad_needed = cfg.pred_horizon_days + 20.0
    if cfg.nominal_padding_days < pad_needed:
        # This only affects cached nominal; UI suggests reset if changed.
        pass

    dt_yr = cfg.dt_days / 365.25
    steps = int(max(2, round(cfg.pred_horizon_days / cfg.dt_days)))

    x = state.x.copy()
    t_yr = state.t_yr

    t_out = np.zeros(steps, dtype=float)
    e_out = np.zeros(steps, dtype=float)
    r_out = np.zeros((steps, 2), dtype=float)

    for i in range(steps):
        td = t_yr * 365.25
        r_nom = nominal_at_time(nom, cfg, td)

        t_out[i] = td
        r_out[i] = x[:2]
        e_out[i] = norm(r_out[i] - r_nom)

        x = rk4_step(x, beta_deg=state.beta_deg, a0=cfg.a0, dt=dt_yr)
        t_yr += dt_yr

    return t_out, e_out, r_out


# -----------------------------
# Scoring
# -----------------------------
def compute_score(state: SimState, cfg: Config) -> Tuple[float, Dict[str, float]]:
    t = state.t_days
    err = state.orbit_err
    pwr = state.power

    within = [1.0 if e <= cfg.err_ok else 0.0 for e in err]
    in_band_days = trapz_integral(within, t)

    track_reward = 6.0 * in_band_days

    err_int = trapz_integral(err, t)
    err_pen = 60.0 * err_int

    low = [1.0 if p < cfg.power_ok else 0.0 for p in pwr]
    low_power_days = trapz_integral(low, t)
    power_pen = 15.0 * low_power_days

    fuel_used = cfg.fuel_start - state.fuel
    fuel_pen = 1.2 * max(0.0, fuel_used)

    dv_bonus = min(25.0, state.dv_mps * 0.06)

    score = max(0.0, track_reward + dv_bonus - err_pen - power_pen - fuel_pen)

    breakdown = dict(
        score=score,
        in_band_days=in_band_days,
        err_integral_AUday=err_int,
        low_power_days=low_power_days,
        fuel_used=fuel_used,
        dv_mps=state.dv_mps,
        track_reward=track_reward,
        err_pen=err_pen,
        power_pen=power_pen,
        fuel_pen=fuel_pen,
        dv_bonus=dv_bonus,
    )
    return score, breakdown


# -----------------------------
# Vega (Altair) charts
# -----------------------------
def comm_rect_df(state: SimState, cfg: Config) -> List[Dict[str, float]]:
    if len(state.t_days) < 2:
        return []
    active = [ea >= cfg.comm_angle_deg for ea in state.earth_angle]
    rows = []
    start = None
    for i, a in enumerate(active):
        if a and start is None:
            start = state.t_days[i]
        if (not a) and start is not None:
            end = state.t_days[i]
            rows.append({"x0": float(start), "x1": float(end)})
            start = None
    if start is not None:
        rows.append({"x0": float(start), "x1": float(state.t_days[-1])})
    return rows


def alt_error_chart(state: SimState, cfg: Config, t_pred: np.ndarray, e_pred: np.ndarray, height: int = 380) -> alt.Chart:
    rows = []
    rows += [{"day": float(d), "value": float(v), "series": "誤差（実）"} for d, v in zip(state.t_days, state.orbit_err)]
    rows += [{"day": float(d), "value": float(v), "series": "誤差（予測:β固定）"} for d, v in zip(t_pred, e_pred)]
    if state.t_days:
        rows += [{"day": float(state.t_days[0]), "value": 0.0, "series": "ノミナル(0)"},
                 {"day": float(max(state.t_days[-1], float(t_pred[-1]))), "value": 0.0, "series": "ノミナル(0)"}]

    x_min = float(min(state.t_days[0], float(t_pred[0]))) if state.t_days else 0.0
    x_max = float(max(state.t_days[-1], float(t_pred[-1]))) if state.t_days else cfg.time_limit_days

    # tolerance band 0..err_ok
    band_df = [{"day": x_min, "y0": 0.0, "y1": cfg.err_ok}, {"day": x_max, "y0": 0.0, "y1": cfg.err_ok}]
    band = alt.Chart(alt.Data(values=band_df)).mark_area(opacity=0.12).encode(
        x=alt.X("day:Q", title="日数"),
        y=alt.Y("y0:Q", title="軌道誤差（AU）"),
        y2="y1:Q",
    )

    # comm windows background
    rects = comm_rect_df(state, cfg)
    comm_layer = alt.Chart(alt.Data(values=rects)).mark_rect(opacity=0.08).encode(
        x="x0:Q", x2="x1:Q",
        y=alt.value(0), y2=alt.value(1)
    )

    line = alt.Chart(alt.Data(values=rows)).mark_line().encode(
        x=alt.X("day:Q", title="日数"),
        y=alt.Y("value:Q", title="軌道誤差（AU）"),
        color=alt.Color("series:N", legend=alt.Legend(title="")),
        strokeDash=alt.StrokeDash(
            "series:N",
            legend=None,  # prevent duplicate legend
            scale=alt.Scale(
                domain=["誤差（実）", "誤差（予測:β固定）", "ノミナル(0)"],
                range=[[1, 0], [6, 3], [2, 2]]
            )
        )
    )

    return (band + comm_layer + line).properties(height=height).configure_axis(labelFontSize=12, titleFontSize=12).configure_legend(labelFontSize=12)


def alt_series(t: List[float], y: List[float], ytitle: str, rule: float | None = None, height: int = 130) -> alt.Chart:
    df = [{"day": float(tt), "value": float(vv)} for tt, vv in zip(t, y)]
    base = alt.Chart(alt.Data(values=df)).mark_line().encode(
        x=alt.X("day:Q", title="日数"),
        y=alt.Y("value:Q", title=ytitle),
        tooltip=[alt.Tooltip("day:Q", title="day"), alt.Tooltip("value:Q", title=ytitle)],
    ).properties(height=height)
    if rule is not None:
        r = alt.Chart(alt.Data(values=[{"value": float(rule)}])).mark_rule(opacity=0.4).encode(y="value:Q")
        return base + r
    return base


def alt_beta(t: List[float], beta: List[float], height: int = 130) -> alt.Chart:
    df = [{"day": float(tt), "beta": float(bb)} for tt, bb in zip(t, beta)]
    return alt.Chart(alt.Data(values=df)).mark_line().encode(
        x=alt.X("day:Q", title="日数"),
        y=alt.Y("beta:Q", title="β角（deg）"),
        tooltip=[alt.Tooltip("day:Q", title="day"), alt.Tooltip("beta:Q", title="β(deg)")],
    ).properties(height=height)


def alt_orbit_chart(state: SimState, cfg: Config, nom: Dict[str, np.ndarray], t_pred: np.ndarray, r_pred: np.ndarray, height: int = 330) -> alt.Chart:
    # Choose time range up to max(pred end, current)
    t_end = float(max(state.t_days[-1] if state.t_days else 0.0, float(t_pred[-1])))
    # nominal subset
    t_nom = nom["t_days"]
    r_nom = nom["r"]
    idx_end = int(clamp(int(round(t_end / cfg.dt_days)), 0, len(t_nom) - 1))

    # Downsample to keep it light
    max_pts = 900
    idx_a = downsample_indices(len(state.rx), max_pts)
    idx_n = downsample_indices(idx_end + 1, max_pts)
    idx_p = downsample_indices(len(r_pred), max_pts)

    rows = []
    for i in idx_a:
        rows.append({"x": float(state.rx[i]), "y": float(state.ry[i]), "series": "実（軌道）"})
    for i in idx_n:
        rows.append({"x": float(r_nom[i,0]), "y": float(r_nom[i,1]), "series": "ノミナル（軌道）"})
    for i in idx_p:
        rows.append({"x": float(r_pred[i,0]), "y": float(r_pred[i,1]), "series": "予測（β固定）"})

    # Orbits (Earth/Venus circles)
    th = np.linspace(0, 2*math.pi, 180)
    for a in th[::2]:
        rows.append({"x": float(R_E*math.cos(a)), "y": float(R_E*math.sin(a)), "series": "地球軌道"})
        rows.append({"x": float(R_V*math.cos(a)), "y": float(R_V*math.sin(a)), "series": "金星軌道"})

    # Current body positions
    r_e = planet_pos(R_E, N_E, (state.t_days[-1]/365.25) if state.t_days else 0.0, 0.0)
    r_v = planet_pos(R_V, N_V, (state.t_days[-1]/365.25) if state.t_days else 0.0, aligned_venus_phase())
    markers = [
        {"x": 0.0, "y": 0.0, "name": "Sun"},
        {"x": float(r_e[0]), "y": float(r_e[1]), "name": "Earth"},
        {"x": float(r_v[0]), "y": float(r_v[1]), "name": "Venus"},
    ]

    domain = 1.35
    base = alt.Chart(alt.Data(values=rows)).mark_line().encode(
        x=alt.X("x:Q", title="x (AU)", scale=alt.Scale(domain=[-domain, domain])),
        y=alt.Y("y:Q", title="y (AU)", scale=alt.Scale(domain=[-domain, domain])),
        color=alt.Color("series:N", legend=alt.Legend(title="")),
        strokeDash=alt.StrokeDash(
            "series:N",
            legend=None,
            scale=alt.Scale(
                domain=["実（軌道）","ノミナル（軌道）","予測（β固定）","地球軌道","金星軌道"],
                range=[[1,0],[6,3],[2,2],[4,4],[4,4]]
            )
        ),
        opacity=alt.Opacity("series:N", legend=None,
                            scale=alt.Scale(domain=["実（軌道）","ノミナル（軌道）","予測（β固定）","地球軌道","金星軌道"],
                                            range=[1.0,0.9,0.9,0.35,0.35])),
        tooltip=[alt.Tooltip("series:N"), alt.Tooltip("x:Q", format=".3f"), alt.Tooltip("y:Q", format=".3f")]
    ).properties(height=height)

    pts = alt.Chart(alt.Data(values=markers)).mark_point(size=80).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[-domain, domain])),
        y=alt.Y("y:Q", scale=alt.Scale(domain=[-domain, domain])),
        tooltip=[alt.Tooltip("name:N"), alt.Tooltip("x:Q", format=".3f"), alt.Tooltip("y:Q", format=".3f")]
    )
    return (base + pts).configure_axis(labelFontSize=12, titleFontSize=12).configure_legend(labelFontSize=12)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="IKAROS β-GO! (Vega v5)", layout="wide")
st.markdown("""<style>.block-container{padding-top:0.8rem;padding-bottom:0.8rem;}</style>""", unsafe_allow_html=True)

st.title("☀️ IKAROS β-GO!（全部Vega v5）")
st.caption("メインは軌道誤差。加えて“軌道（2D）”も戻した版。SRPの小ささは μm/s² と ΔV で見える化。")

with st.sidebar:
    st.header("設定")
    hard_mode = st.toggle("高難易度（通信中だけβ変更＋燃料制限）", value=False)
    show_teacher = st.toggle("先生モード", value=False)
    st.divider()

    cfg = Config()
    cfg.hard_mode = bool(hard_mode)

    st.subheader("更新（重い時はここ）")
    cfg.refresh_ms = st.slider("更新間隔(ms)", 350, 2500, cfg.refresh_ms, 50)
    cfg.ticks_per_refresh = st.slider("1回で進める回数", 1, 6, cfg.ticks_per_refresh, 1)

    st.divider()
    st.subheader("ゲームバランス（初期誤差）")
    cfg.init_pos_offset_au = st.slider("初期 位置誤差(AU)", 0.0, 0.02, cfg.init_pos_offset_au, 0.001)
    cfg.init_vel_offset_auyr = st.slider("初期 速度誤差(AU/yr)", 0.0, 0.02, cfg.init_vel_offset_auyr, 0.0005)
    cfg.init_beta_deg = st.slider("初期 β(deg)", -45.0, 45.0, cfg.init_beta_deg, 1.0)

    if show_teacher:
        st.divider()
        st.subheader("先生モード：モデル/評価")
        cfg.dt_days = st.slider("刻み(days)", 0.2, 2.0, cfg.dt_days, 0.1)
        cfg.time_limit_days = st.slider("制限時間(days)", 140.0, 420.0, cfg.time_limit_days, 5.0)
        cfg.pred_horizon_days = st.slider("予測ホライズン(days)", 20.0, 180.0, cfg.pred_horizon_days, 5.0)
        cfg.nominal_padding_days = st.slider("ノミナル余白(days)", 80.0, 320.0, cfg.nominal_padding_days, 10.0)
        cfg.a0 = st.slider("SRP強さ(AU/yr^2)", 0.005, 0.06, cfg.a0, 0.001)
        cfg.comm_angle_deg = st.slider("通信地球角しきい値(deg)", 5.0, 30.0, cfg.comm_angle_deg, 1.0)
        cfg.power_ok = st.slider("発電OK閾値", 0.2, 1.2, cfg.power_ok, 0.05)
        cfg.err_ok = st.slider("許容誤差帯(AU)", 0.002, 0.05, cfg.err_ok, 0.001)
        cfg.fuel_start = st.slider("燃料（姿勢リソース）", 40.0, 240.0, cfg.fuel_start, 5.0)
        cfg.fuel_per_deg = st.slider("燃料/deg", 0.05, 1.0, cfg.fuel_per_deg, 0.05)

# Fix for prediction weirdness when teacher sets too small padding:
# we enforce a minimum padding relative to pred horizon (note: cached nominal depends on it).
cfg.nominal_padding_days = max(cfg.nominal_padding_days, cfg.pred_horizon_days + 30.0)

# Cached nominal
nominal = simulate_nominal_cached(cfg.dt_days, cfg.time_limit_days, cfg.nominal_padding_days, cfg.a0)

# Session init
if "sim_v5" not in st.session_state:
    st.session_state.sim_v5 = init_state(cfg)
    st.session_state.auto_v5 = True

state: SimState = st.session_state.sim_v5

def rerun():
    (st.rerun() if hasattr(st, "rerun") else st.experimental_rerun())

def reset_all():
    st.session_state.sim_v5 = init_state(cfg)
    st.session_state.auto_v5 = True
    rerun()

# Controls
c1, c2, c3, c4, c5 = st.columns([1.0, 1.0, 1.0, 1.25, 1.25], gap="small")
with c1:
    if st.button("🔁 リセット", use_container_width=True):
        reset_all()

comm_ok = comm_window_active(state.x[:2], state.t_yr, cfg)
lock_reason = None
if cfg.hard_mode and not comm_ok:
    lock_reason = "通信ウィンドウ外"
if cfg.hard_mode and state.fuel <= 0:
    lock_reason = "燃料切れ"

def apply_beta_change(new_beta: float):
    if cfg.hard_mode and lock_reason is not None:
        return
    new_beta = float(clamp(new_beta, -75.0, 75.0))
    d = abs(new_beta - state.beta_deg)
    cost = d * cfg.fuel_per_deg
    if cfg.hard_mode and state.fuel - cost < 0:
        return
    state.beta_deg = new_beta
    state.fuel = max(0.0, state.fuel - cost)

with c2:
    if st.button("⬅ β -5°", use_container_width=True, disabled=(cfg.hard_mode and lock_reason is not None)):
        apply_beta_change(state.beta_deg - 5.0)
with c3:
    if st.button("➡ β +5°", use_container_width=True, disabled=(cfg.hard_mode and lock_reason is not None)):
        apply_beta_change(state.beta_deg + 5.0)
with c4:
    preset = st.selectbox("ワンタッチβ", ["-35°（内側へ）", "-20°（ノミナル寄り）", "0°（待ち）", "+20°（外側へ）"], index=2)
    if preset.startswith("-35"):
        apply_beta_change(-35.0)
    elif preset.startswith("-20"):
        apply_beta_change(-20.0)
    elif preset.startswith("0"):
        apply_beta_change(0.0)
    else:
        apply_beta_change(20.0)
with c5:
    a, b = st.columns(2)
    with a:
        if st.button("⏸/▶ 自動進行", use_container_width=True):
            st.session_state.auto_v5 = not st.session_state.auto_v5
    with b:
        if st.button("▶ 進める（約5日）", use_container_width=True):
            st.session_state.manual_v5 = True

auto = st.session_state.auto_v5
manual = st.session_state.get("manual_v5", False)
st.session_state.manual_v5 = False

# HUD (always)
t_days_now = state.t_yr * 365.25
r_sc = state.x[:2]
pwr_now = power_frac(r_sc, state.beta_deg)
ea_now = earth_angle_deg(r_sc, state.t_yr)
r_nom_now = nominal_at_time(nominal, cfg, t_days_now)
err_now = norm(r_sc - r_nom_now)

a_vec = srp_accel(r_sc, state.beta_deg, cfg.a0)
a_mps2 = norm(a_vec) * AUYR2_TO_MPS2
state.accel_ums2 = a_mps2 * 1e6

m1, m2, m3, m4, m5, m6 = st.columns([1,1,1,1,1,1], gap="small")
m1.metric("β角", f"{state.beta_deg:+.0f}°")
m2.metric("発電量", f"{pwr_now:.2f}", delta=("OK" if pwr_now >= cfg.power_ok else "不足"))
m3.metric("地球角", f"{ea_now:.1f}°", delta=("COMMS" if comm_ok else "NO-COMMS"))
m4.metric("軌道誤差", f"{err_now:.3f} AU", delta=(f"帯≤{cfg.err_ok:.3f}" if err_now <= cfg.err_ok else "帯の外"))
m5.metric("燃料", f"{state.fuel:.1f}")
m6.metric("SRP加速度", f"{state.accel_ums2:.2f} μm/s²", help="SRPはμm/s²オーダー。大きなΔVは期待できない。")

if cfg.hard_mode and lock_reason is not None:
    st.warning(f"β変更ロック中（高難易度）：{lock_reason}")

# Logging
def log_snapshot(td: float, r: np.ndarray, p: float, ea: float, err: float, dv: float, acc_ums2: float):
    state.t_days.append(float(td))
    state.rx.append(float(r[0]))
    state.ry.append(float(r[1]))
    state.beta.append(float(state.beta_deg))
    state.power.append(float(p))
    state.earth_angle.append(float(ea))
    state.orbit_err.append(float(err))
    state.fuel_log.append(float(state.fuel))
    state.dv_log.append(float(dv))
    state.accel_log_ums2.append(float(acc_ums2))

if not state.t_days and state.phase == "play":
    log_snapshot(t_days_now, r_sc, pwr_now, ea_now, err_now, state.dv_mps, state.accel_ums2)

# Integrate
def integrate_steps(n_steps: int):
    dt_yr = cfg.dt_days / 365.25
    for _ in range(n_steps):
        if state.t_yr * 365.25 >= cfg.time_limit_days:
            state.phase = "result"
            break

        # ΔV: integrate |a| dt (SRP-only)
        r = state.x[:2]
        a = srp_accel(r, state.beta_deg, cfg.a0)
        a_mps2_local = norm(a) * AUYR2_TO_MPS2
        dt_s = dt_yr * YR_S
        state.dv_mps += a_mps2_local * dt_s

        # integrate dynamics
        state.x = rk4_step(state.x, beta_deg=state.beta_deg, a0=cfg.a0, dt=dt_yr)
        state.t_yr += dt_yr

        td = state.t_yr * 365.25
        r_sc2 = state.x[:2]
        p2 = power_frac(r_sc2, state.beta_deg)
        ea2 = earth_angle_deg(r_sc2, state.t_yr)
        r_nom2 = nominal_at_time(nominal, cfg, td)
        err2 = norm(r_sc2 - r_nom2)
        a2 = srp_accel(r_sc2, state.beta_deg, cfg.a0)
        acc2_ums2 = norm(a2) * AUYR2_TO_MPS2 * 1e6

        log_snapshot(td, r_sc2, p2, ea2, err2, state.dv_mps, acc2_ums2)

# Advance
if state.phase == "play":
    if auto and HAVE_AUTOREFRESH:
        st_autorefresh(interval=cfg.refresh_ms, key="tick_v5")
        integrate_steps(cfg.ticks_per_refresh)
    elif manual:
        integrate_steps(int(max(1, round(5.0 / cfg.dt_days))))

# Prediction (error + orbit)
t_pred, e_pred, r_pred = predict_future(state, cfg, nominal)

# Layout: error + orbit (left), live (right)
L, R = st.columns([1.60, 1.0], gap="large")

with L:
    st.subheader("軌道誤差（メイン）")
    st.caption("実=実際 / 予測=いまのβ固定 / ノミナル=0 / 薄帯=許容誤差 / 背景薄帯=通信区間")
    st.altair_chart(alt_error_chart(state, cfg, t_pred, e_pred), use_container_width=True)

    st.subheader("軌道（2D）")
    st.caption("実（軌道）/ ノミナル（軌道）/ 予測（β固定）に加え、地球軌道・金星軌道も表示")
    st.altair_chart(alt_orbit_chart(state, cfg, nominal, t_pred, r_pred), use_container_width=True)

with R:
    st.subheader("ライブ（全部Vega）")
    st.caption("発電量（相対）— 閾値より下は運用がきつい")
    st.altair_chart(alt_series(state.t_days, state.power, "power", rule=cfg.power_ok), use_container_width=True)

    st.caption("地球角（通信のしやすさ）— しきい値以上が通信ウィンドウ")
    st.altair_chart(alt_series(state.t_days, state.earth_angle, "deg", rule=cfg.comm_angle_deg), use_container_width=True)

    st.caption("SRPの効きの小ささ：ΔV（積分, m/s）")
    st.altair_chart(alt_series(state.t_days, state.dv_log, "m/s"), use_container_width=True)

    st.caption("SRP加速度（μm/s²）")
    st.altair_chart(alt_series(state.t_days, state.accel_log_ums2, "μm/s²"), use_container_width=True)

    st.caption("燃料（姿勢リソース）")
    st.altair_chart(alt_series(state.t_days, state.fuel_log, "fuel"), use_container_width=True)

    st.caption("β角（操作履歴）")
    st.altair_chart(alt_beta(state.t_days, state.beta), use_container_width=True)

# Result screen
if state.phase == "result":
    st.divider()
    st.header("📊 リザルト（Vega）")

    score, breakdown = compute_score(state, cfg)
    st.subheader(f"スコア：{score:.0f} 点")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("ΔV相当", f"{breakdown['dv_mps']:.0f} m/s")
    k2.metric("帯の中にいた日数", f"{breakdown['in_band_days']:.1f} d")
    k3.metric("低発電日数", f"{breakdown['low_power_days']:.1f} d")
    k4.metric("燃料使用", f"{breakdown['fuel_used']:.1f}")

    with st.expander("スコア内訳", expanded=False):
        st.write(breakdown)

    st.subheader("結果グラフ")
    st.altair_chart(alt_error_chart(state, cfg, t_pred, e_pred, height=440), use_container_width=True)
    st.altair_chart(alt_orbit_chart(state, cfg, nominal, t_pred, r_pred, height=380), use_container_width=True)

    r1, r2 = st.columns(2, gap="large")
    with r1:
        st.altair_chart(alt_series(state.t_days, state.power, "power", rule=cfg.power_ok, height=220).properties(title="発電量"), use_container_width=True)
        st.altair_chart(alt_series(state.t_days, state.dv_log, "m/s", height=220).properties(title="ΔV（SRPの小ささ）"), use_container_width=True)
        st.altair_chart(alt_beta(state.t_days, state.beta, height=220).properties(title="β角"), use_container_width=True)
    with r2:
        st.altair_chart(alt_series(state.t_days, state.earth_angle, "deg", rule=cfg.comm_angle_deg, height=220).properties(title="地球角"), use_container_width=True)
        st.altair_chart(alt_series(state.t_days, state.fuel_log, "fuel", height=220).properties(title="燃料（姿勢リソース）"), use_container_width=True)
        st.altair_chart(alt_series(state.t_days, state.orbit_err, "AU", height=220).properties(title="誤差（実）"), use_container_width=True)

    st.divider()
    if st.button("もう一回（リセット）", use_container_width=True):
        reset_all()

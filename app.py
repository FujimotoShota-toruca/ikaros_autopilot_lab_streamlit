\
# app.py
# IKAROS-GO (prototype) — a lightweight B-plane guidance “ops experience” game
#
# How to use:
#   streamlit run app.py
#
# You can later replace the toy schedules with real precomputed tables:
#   - data/angles_schedule.json
#   - data/sensitivity_schedule.json
#
# See docs/math.md and docs/customize.md for details.

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="IKAROS-GO (prototype)", layout="wide")


# -----------------------------
# Configuration
# -----------------------------

DATA_DIR = Path(__file__).parent / "data"


@dataclass
class GameConfig:
    n_turns: int = 14              # 14 turns * 2 weeks = 28 weeks
    turn_days: int = 14
    target_bt_br_km: Tuple[float, float] = (0.0, 0.0)
    tolerance_km: float = 30.0

    # Constraints (toy defaults)
    sun_angle_max_deg: float = 45.0

    # Slider bounds for delta-beta controls
    beta_step_max_deg: float = 15.0
    beta_step_res_deg: float = 0.5

    # Noise levels (toy)
    process_noise_km: float = 3.0
    measurement_noise_km: float = 6.0
    init_estimate_noise_km: float = 10.0


CFG = GameConfig()


# -----------------------------
# Helpers: constraints & schedules
# -----------------------------

def comm_ok(earth_aspect_angle_deg: float) -> bool:
    """
    Communication constraint (simplified):
      Earth aspect angle within [0, 60] or [120, 180] degrees => comm possible.
    """
    a = abs(earth_aspect_angle_deg) % 360.0
    a = a if a <= 180.0 else 360.0 - a
    return (0.0 <= a <= 60.0) or (120.0 <= a <= 180.0)


def no_link_event(day: float) -> bool:
    """
    Toy blackout window to emulate unavoidable no-link periods.
    Replace/remove if you use a real schedule.
    """
    return 100.0 <= day <= 130.0


def load_angles_schedule() -> Optional[List[Dict[str, float]]]:
    """
    Optional: load nominal sun/earth angles from JSON (one entry per turn).
    Format:
      [
        {"turn": 0, "day": 0, "sun_angle_deg": 35.0, "earth_angle_deg": 120.0},
        ...
      ]
    """
    p = DATA_DIR / "angles_schedule.json"
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return None
    return data


def load_sensitivity_schedule() -> Optional[List[Dict[str, object]]]:
    """
    Optional: load sensitivity matrices C_k from JSON (one entry per turn).
    Format:
      [
        {"turn": 0, "C": [[1.0, 0.3], [-0.2, 0.9]]},
        ...
      ]
    """
    p = DATA_DIR / "sensitivity_schedule.json"
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return None
    return data


ANGLES_SCHEDULE = load_angles_schedule()
SENS_SCHEDULE = load_sensitivity_schedule()


def nominal_angles(day: float) -> Tuple[float, float]:
    """
    Returns (sun_angle_deg, earth_aspect_angle_deg) for the current day.
    If angles_schedule.json exists, uses it by nearest turn; otherwise uses toy sinusoids.
    """
    if ANGLES_SCHEDULE:
        # Find nearest by day (or use turn index if present)
        # Since the game is turn-based, we'll use the closest entry by day.
        best = min(ANGLES_SCHEDULE, key=lambda d: abs(float(d.get("day", 0.0)) - day))
        return float(best["sun_angle_deg"]), float(best["earth_angle_deg"])

    # Toy schedules (replace with your precomputed timeline)
    sun = 35.0 + 8.0 * np.sin(2.0 * np.pi * day / 180.0)
    earth = 90.0 + 70.0 * np.sin(2.0 * np.pi * day / 120.0)
    return float(sun), float(earth)


def make_Ck(turn: int, k_true: float) -> np.ndarray:
    """
    Sensitivity matrix mapping u=[Δβ_in, Δβ_out] to x=[ΔB_T, ΔB_R] (toy).
    If sensitivity_schedule.json exists, uses it and scales by k_true.
    """
    if SENS_SCHEDULE:
        # Match by "turn"
        entry = next((d for d in SENS_SCHEDULE if int(d.get("turn", -1)) == int(turn)), None)
        if entry is not None:
            C = np.array(entry["C"], dtype=float)
            return k_true * C

    # Toy: controllability improves in later turns
    gain = (0.6 + 0.06 * turn) * k_true
    return gain * np.array([[1.0, 0.3],
                            [-0.2, 0.9]], dtype=float)


# -----------------------------
# Session state
# -----------------------------

def init_state(seed: int = 2) -> None:
    rng = np.random.default_rng(seed)

    st.session_state.k = 0
    st.session_state.day = 0.0

    # True & estimated state on B-plane (km)
    st.session_state.x_true = np.array([120.0, -80.0], dtype=float)
    st.session_state.x_hat = st.session_state.x_true + rng.normal(0.0, CFG.init_estimate_noise_km, size=2)

    # Hidden sail efficiency and its estimate
    st.session_state.k_true = float(rng.uniform(0.75, 1.25))
    st.session_state.k_hat = 1.0

    st.session_state.hist = []  # list of dict logs
    st.session_state.rng_seed = seed


if "k" not in st.session_state:
    init_state()


# -----------------------------
# UI
# -----------------------------

st.title("IKAROS-GO (prototype) — B-plane guidance with ops constraints")

with st.sidebar:
    st.header("Settings")
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=int(st.session_state.rng_seed), step=1)
    if st.button("Reset mission"):
        init_state(int(seed))
        st.rerun()

    st.markdown("---")
    st.caption("Tip: put real schedules in data/*.json to replace the toy model.")
    st.caption("See docs/customize.md.")

colL, colR = st.columns([1.2, 0.8], gap="large")

# Current constraints
sun_angle, earth_angle = nominal_angles(st.session_state.day)
link_ok = (not no_link_event(st.session_state.day)) and (sun_angle <= CFG.sun_angle_max_deg) and comm_ok(earth_angle)

# Control panel
with colR:
    st.subheader("Control")
    st.caption(f"Turn {st.session_state.k + 1}/{CFG.n_turns}  |  Day {st.session_state.day:.0f}")

    c1, c2 = st.columns(2)
    c1.metric("Sun angle (deg)", f"{sun_angle:.1f}", help=f"Power constraint: must be < {CFG.sun_angle_max_deg:.0f}°")
    c2.metric("Earth aspect angle (deg)", f"{earth_angle:.1f}", help="Comms when 0–60° or 120–180° (simplified)")

    st.metric("LINK", "OK ✅" if link_ok else "NO ❌")

    beta_in = st.slider("Δβ_in (deg)", -CFG.beta_step_max_deg, CFG.beta_step_max_deg, 0.0, CFG.beta_step_res_deg,
                        disabled=not link_ok)
    beta_out = st.slider("Δβ_out (deg)", -CFG.beta_step_max_deg, CFG.beta_step_max_deg, 0.0, CFG.beta_step_res_deg,
                         disabled=not link_ok)

    step = st.button("Execute 2-week cycle", type="primary", disabled=(st.session_state.k >= CFG.n_turns))

    # Dynamics step
    if step:
        rng = np.random.default_rng(int(st.session_state.rng_seed) + int(st.session_state.k) + 100)

        u = np.array([beta_in, beta_out], dtype=float) if link_ok else np.array([0.0, 0.0], dtype=float)
        Ck = make_Ck(st.session_state.k, float(st.session_state.k_true))
        w = rng.normal(0.0, CFG.process_noise_km, size=2)

        # True propagation on B-plane (toy)
        st.session_state.x_true = st.session_state.x_true + Ck @ u + w

        # OD update only if we have comm link
        if link_ok:
            meas = st.session_state.x_true + rng.normal(0.0, CFG.measurement_noise_km, size=2)
            st.session_state.x_hat = 0.6 * st.session_state.x_hat + 0.4 * meas
            st.session_state.k_hat += 0.15 * (float(st.session_state.k_true) - float(st.session_state.k_hat))

        # Log
        st.session_state.hist.append({
            "turn": int(st.session_state.k),
            "day": float(st.session_state.day),
            "u_in_deg": float(u[0]),
            "u_out_deg": float(u[1]),
            "BT_hat_km": float(st.session_state.x_hat[0]),
            "BR_hat_km": float(st.session_state.x_hat[1]),
            "link": bool(link_ok),
            "sun_deg": float(sun_angle),
            "earth_deg": float(earth_angle),
            "k_hat": float(st.session_state.k_hat),
        })

        # Advance time
        st.session_state.k += 1
        st.session_state.day += float(CFG.turn_days)

        if st.session_state.k >= CFG.n_turns:
            st.success("Mission end! (prototype)")

        st.rerun()


# B-plane plot
with colL:
    st.subheader("B-plane")

    target = np.array(CFG.target_bt_br_km, dtype=float)
    tol = float(CFG.tolerance_km)

    fig = go.Figure()

    # Target tolerance circle (toy)
    th = np.linspace(0, 2 * np.pi, 240)
    fig.add_trace(go.Scatter(
        x=target[0] + tol * np.cos(th),
        y=target[1] + tol * np.sin(th),
        mode="lines",
        name="target tolerance",
    ))

    # Past estimated path
    if st.session_state.hist:
        xs = [h["BT_hat_km"] for h in st.session_state.hist]
        ys = [h["BR_hat_km"] for h in st.session_state.hist]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name="estimate path"))

    # Current estimate
    fig.add_trace(go.Scatter(
        x=[float(st.session_state.x_hat[0])],
        y=[float(st.session_state.x_hat[1])],
        mode="markers",
        name="current estimate",
    ))

    fig.update_layout(
        xaxis_title="B_T (km)",
        yaxis_title="B_R (km)",
        height=560,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=True,
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Log")
st.dataframe(st.session_state.hist, use_container_width=True)

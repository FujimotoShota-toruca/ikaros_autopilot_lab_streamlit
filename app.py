# IKAROS：B-plane ダーツ（運用ゲーム） v11
# Fixes from v10:
# - Remove japanize_matplotlib (breaks on Python 3.13 due to distutils removal)
# - Robust Japanese font setup using installed NotoSansCJK (if available)
# - Restore score_game() (v10 regression)
# - Keep: B-plane main, sliders only, nominal meets Venus at arrival, 3D geometry

from __future__ import annotations

import os
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Polygon, Circle, Ellipse
from matplotlib import font_manager as fm

import plotly.graph_objects as go


# -----------------------------
# Matplotlib Japanese font setup (no extra deps)
# -----------------------------
def setup_matplotlib_japanese() -> Tuple[Optional[str], Optional[str]]:
    """
    Try to configure a Japanese-capable font on Streamlit Cloud / Linux.
    Prefers Noto Sans CJK JP (often preinstalled). Falls back gracefully.
    """
    # Known common locations (Streamlit Cloud often has NotoSansCJK*.ttc)
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf",
        "/usr/share/fonts/opentype/noto/NotoSansJP-Regular.otf",
        "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
        "/usr/share/fonts/truetype/ipafont-gothic/ipag.ttf",
        "/usr/share/fonts/truetype/takao-gothic/TakaoGothic.ttf",
    ]

    # Search system fonts too (robust to different OS images)
    try:
        for ext in ("ttf", "otf", "ttc"):
            for p in fm.findSystemFonts(fontpaths=None, fontext=ext):
                base = os.path.basename(p).lower()
                if any(k in base for k in ["notosanscjk", "notosansjp", "sourcehansans", "ipag", "ipa", "takaogothic"]):
                    candidates.append(p)
    except Exception:
        pass

    seen = set()
    for path in candidates:
        if not path or path in seen:
            continue
        seen.add(path)
        if os.path.exists(path):
            try:
                fm.fontManager.addfont(path)
                name = fm.FontProperties(fname=path).get_name()
                matplotlib.rcParams["font.family"] = name
                matplotlib.rcParams["axes.unicode_minus"] = False
                return name, path
            except Exception:
                continue

    # Fallback (may show tofu for Japanese)
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    matplotlib.rcParams["axes.unicode_minus"] = False
    return None, None


JP_FONT_NAME, JP_FONT_PATH = setup_matplotlib_japanese()


# -----------------------------
# Utils
# -----------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def wrap180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0

def l2(xy: np.ndarray) -> float:
    return float(np.linalg.norm(xy))

def cosd(deg: float) -> float:
    return math.cos(math.radians(deg))

def fmt_bool(b: bool) -> str:
    return "🟢OK" if b else "🔴NG"


# -----------------------------
# Ephemeris (very simplified)
# -----------------------------
@dataclass
class EphemConfig:
    omega_earth: float = 360.0 / 365.25
    omega_venus: float = 360.0 / 224.7
    r_earth: float = 1.0
    r_venus: float = 0.723
    theta_e0: float = 0.0
    theta_v0: float = 35.0
    t_end_day: float = 170.0  # last section time

def pos_au(r: float, theta_deg: float) -> np.ndarray:
    th = math.radians(theta_deg)
    return np.array([r * math.cos(th), r * math.sin(th)], dtype=float)

def ephem_at_day(t_day: float, eph: EphemConfig) -> Dict[str, np.ndarray]:
    th_e = eph.theta_e0 + eph.omega_earth * t_day
    th_v = eph.theta_v0 + eph.omega_venus * t_day
    return {"earth": pos_au(eph.r_earth, th_e), "venus": pos_au(eph.r_venus, th_v), "th_e": th_e, "th_v": th_v}

def sc_nominal_at_index(k: int, n: int, eph: EphemConfig) -> Dict[str, np.ndarray]:
    """
    “ノミナル（計画）”は最終インデックスで金星位置(到着時刻)に一致させる。
    角度：Earth(t=0) -> Venus(t=t_end) を線形補間（概念）
    半径：1AU -> 0.723AU を線形補間（概念）
    """
    frac = 0.0 if n <= 1 else k / (n - 1)

    e0 = ephem_at_day(0.0, eph)
    vend = ephem_at_day(eph.t_end_day, eph)

    th_start = float(e0["th_e"])
    th_target = float(vend["th_v"])

    th_sc = (1.0 - frac) * th_start + frac * th_target
    r_sc = eph.r_earth + (eph.r_venus - eph.r_earth) * frac
    p_sc = pos_au(r_sc, th_sc)
    return {"sc": p_sc, "th_sc": th_sc, "r_sc": r_sc}

def earth_angle_from_geometry(t_day: float, k: int, n: int, eph: EphemConfig) -> float:
    """Signed heliocentric phase angle (Earth - SC) in degrees."""
    e = ephem_at_day(t_day, eph)
    sc = sc_nominal_at_index(k, n, eph)
    return float(wrap180(float(e["th_e"]) - float(sc["th_sc"])))


# -----------------------------
# Game model
# -----------------------------
@dataclass
class Section:
    name: str
    t_day: float
    S: np.ndarray
    dbeta_in_max: float
    dbeta_out_max: float
    uplink_possible: bool
    maneuvers_per_deg: float
    od_gain: float

def build_sections() -> List[Section]:
    def mat(a, b, c, d):
        return np.array([[a, b], [c, d]], dtype=float)

    times = [0, 25, 55, 85, 115, 145, 170]
    S_pre = mat(180, 40, -20, 140)
    S_pre2 = mat(210, 60, -40, 170)
    S_pre3 = mat(240, 70, -60, 190)
    S_post = mat(520, 130, -90, 430)
    S_post2 = mat(560, 150, -110, 460)
    S_post3 = mat(600, 170, -120, 500)
    S_post4 = mat(640, 190, -140, 520)

    return [
        Section("Section 1", times[0], S_pre, 6, 6, True, 65, 0.45),
        Section("Section 2", times[1], S_pre2, 6, 6, True, 80, 0.50),
        Section("Section 3", times[2], S_pre3, 5, 5, True, 95, 0.55),
        Section("Section 4（NO-LINK）", times[3], S_post, 0, 0, False, 0, 0.60),
        Section("Section 5", times[4], S_post2, 18, 18, True, 45, 0.70),
        Section("Section 6", times[5], S_post3, 18, 18, True, 35, 0.78),
        Section("Section 7（到着）", times[6], S_post4, 15, 15, True, 30, 0.85),
    ]


@dataclass
class GameConfig:
    target: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0], dtype=float))
    target_radius_early_km: float = 9000.0
    target_radius_late_km: float = 2000.0
    target_tighten_section: int = 5

    init_error_sigma_km: float = 6500.0
    init_est_sigma_km: float = 1200.0

    sigma_gain_in0: float = 0.10
    sigma_gain_out0: float = 0.08
    meas_sigma_km: float = 500.0

    rcs_sigma_per_sqrt_maneuver: float = 30.0

    maneuver_budget: float = 6000.0
    plan_beta_in_deg: float = 0.0
    plan_beta_out_deg: float = 0.0

    # comm and pointing model
    comm_window_deg: float = 20.0
    beta_point_coupling: float = 0.70

    # energy
    energy_max: float = 200.0
    energy_init: float = 140.0
    energy_min_for_comm: float = 30.0
    base_load: float = 70.0
    gen_scale: float = 90.0
    comm_cost: float = 10.0
    maneuver_energy_scale: float = 0.02

    # science data
    data_buffer_max: float = 60.0
    data_collect_hi: float = 12.0
    data_collect_lo: float = 4.0
    data_downlink_cap: float = 18.0

    pred_ellipse_sigma: float = 1.0

    eph: EphemConfig = field(default_factory=EphemConfig)


@dataclass
class GameState:
    k: int
    B_est: np.ndarray
    B_true: np.ndarray
    B_obs_last: Optional[np.ndarray]

    p_true: np.ndarray
    p_est: np.ndarray
    P_cov: np.ndarray

    beta_in: float
    beta_out: float
    maneuvers_left: float

    log: List[Dict]
    phase: str
    rng_state: Dict

    energy: float
    data_buffer: float
    data_downlinked: float
    data_lost: float
    blackout_count: int


def init_game(cfg: GameConfig, sections: List[Section], seed: int) -> GameState:
    rng = np.random.default_rng(seed)

    p_true = np.array([1.0 + rng.normal(0, cfg.sigma_gain_in0), 1.0 + rng.normal(0, cfg.sigma_gain_out0)], dtype=float)
    p_est = np.array([1.0, 1.0], dtype=float)
    P_cov = np.diag([cfg.sigma_gain_in0**2, cfg.sigma_gain_out0**2])

    B_true = cfg.target + rng.normal(0, cfg.init_error_sigma_km, size=(2,))
    B_est = B_true + rng.normal(0, cfg.init_est_sigma_km, size=(2,))

    return GameState(
        k=0,
        B_est=B_est,
        B_true=B_true,
        B_obs_last=None,
        p_true=p_true,
        p_est=p_est,
        P_cov=P_cov,
        beta_in=cfg.plan_beta_in_deg,
        beta_out=cfg.plan_beta_out_deg,
        maneuvers_left=cfg.maneuver_budget,
        log=[],
        phase="play",
        rng_state={"seed": seed, "bitgen": rng.bit_generator.state},
        energy=float(cfg.energy_init),
        data_buffer=0.0,
        data_downlinked=0.0,
        data_lost=0.0,
        blackout_count=0,
    )


# -----------------------------
# Ops model
# -----------------------------
def beta_eff(beta_in: float, beta_out: float) -> float:
    return 0.5 * (abs(beta_in) + abs(beta_out))

def beta_pointing(beta_in: float, beta_out: float) -> float:
    return 0.5 * (beta_in + beta_out)

def earth_angle_base_deg(state: GameState, cfg: GameConfig, sections: List[Section]) -> float:
    sec = sections[min(state.k, len(sections) - 1)]
    return earth_angle_from_geometry(sec.t_day, min(state.k, len(sections)-1), len(sections), cfg.eph)

def predicted_earth_angle_deg(beta_in: float, beta_out: float, state: GameState, cfg: GameConfig, sections: List[Section]) -> float:
    base = earth_angle_base_deg(state, cfg, sections)
    return float(wrap180(base + cfg.beta_point_coupling * beta_pointing(beta_in, beta_out)))

def comm_available(beta_in: float, beta_out: float, state: GameState, cfg: GameConfig, sections: List[Section]) -> bool:
    sec = sections[min(state.k, len(sections) - 1)]
    if not sec.uplink_possible:
        return False
    if state.energy < cfg.energy_min_for_comm:
        return False
    ea = predicted_earth_angle_deg(beta_in, beta_out, state, cfg, sections)
    return bool(abs(ea) <= cfg.comm_window_deg)


# -----------------------------
# OD update
# -----------------------------
def od_update_gains(
    B_obs: np.ndarray,
    B_pred: np.ndarray,
    dβ: np.ndarray,
    section: Section,
    state: GameState,
    cfg: GameConfig,
    od_gain_eff: float,
):
    r = B_obs - B_pred
    G = section.S @ np.diag([float(dβ[0]), float(dβ[1])])
    R = np.eye(2) * (cfg.meas_sigma_km**2)
    P = state.P_cov

    S_mat = G @ P @ G.T + R
    try:
        invS = np.linalg.inv(S_mat)
    except np.linalg.LinAlgError:
        invS = np.linalg.pinv(S_mat)

    K = P @ G.T @ invS
    K_eff = od_gain_eff * K

    dp = K_eff @ r
    p_est_new = state.p_est + dp

    I = np.eye(2)
    P_new = (I - K_eff @ G) @ P @ (I - K_eff @ G).T + K_eff @ R @ K_eff.T
    return p_est_new, P_new


# -----------------------------
# Preview + step
# -----------------------------
def clamp_dbeta(dβ: np.ndarray, section: Section) -> np.ndarray:
    d = dβ.copy()
    d[0] = clamp(d[0], -section.dbeta_in_max, section.dbeta_in_max)
    d[1] = clamp(d[1], -section.dbeta_out_max, section.dbeta_out_max)
    return d

def applied_dbeta_and_comm(state: GameState, cfg: GameConfig, sections: List[Section], section: Section) -> Tuple[np.ndarray, bool]:
    plan = np.array([cfg.plan_beta_in_deg, cfg.plan_beta_out_deg], dtype=float)
    cmd = np.array([state.beta_in, state.beta_out], dtype=float)
    dβ = cmd - plan

    comm_ok = comm_available(float(cmd[0]), float(cmd[1]), state, cfg, sections)
    if not comm_ok:
        dβ = np.array([0.0, 0.0], dtype=float)

    dβ = clamp_dbeta(dβ, section)
    return dβ, comm_ok

def scale_by_budget(dβ: np.ndarray, section: Section, maneuvers_left: float) -> Tuple[np.ndarray, float, bool]:
    total_deg = abs(float(dβ[0])) + abs(float(dβ[1]))
    maneuvers = section.maneuvers_per_deg * total_deg
    if maneuvers <= maneuvers_left:
        return dβ, float(maneuvers), False
    if maneuvers_left <= 0:
        return np.array([0.0, 0.0], dtype=float), 0.0, True
    scale = maneuvers_left / max(maneuvers, 1e-9)
    dβ2 = dβ * scale
    maneuvers2 = section.maneuvers_per_deg * (abs(float(dβ2[0])) + abs(float(dβ2[1])))
    return dβ2, float(maneuvers2), True

def preview_next(state: GameState, cfg: GameConfig, sections: List[Section], section: Section) -> Dict:
    dβ0, comm_ok = applied_dbeta_and_comm(state, cfg, sections, section)
    dβ, maneuvers, limited = scale_by_budget(dβ0, section, state.maneuvers_left)

    u_est = np.array([dβ[0] * state.p_est[0], dβ[1] * state.p_est[1]], dtype=float)
    B_pred = state.B_est + section.S @ u_est

    G = section.S @ np.diag([float(dβ[0]), float(dβ[1])])
    cov_gain = G @ state.P_cov @ G.T

    sig_rcs = cfg.rcs_sigma_per_sqrt_maneuver * math.sqrt(max(maneuvers, 0.0))
    cov_rcs = np.eye(2) * (sig_rcs**2)
    cov_pred = cov_gain + cov_rcs

    return {"dβ": dβ, "comm_ok": comm_ok, "maneuvers": maneuvers, "budget_limited": bool(limited), "B_pred": B_pred, "cov_pred": cov_pred}

def execute_section(state: GameState, cfg: GameConfig, sections: List[Section]) -> None:
    rng = np.random.default_rng()
    rng.bit_generator.state = state.rng_state["bitgen"]

    section = sections[state.k]
    pv = preview_next(state, cfg, sections, section)
    dβ = pv["dβ"]
    comm_ok = pv["comm_ok"]
    maneuvers = pv["maneuvers"]

    state.maneuvers_left -= maneuvers

    # Power
    be = beta_eff(state.beta_in, state.beta_out) if comm_ok else 0.0
    gen = cfg.gen_scale * max(0.0, cosd(be))
    cost = cfg.base_load + cfg.maneuver_energy_scale * maneuvers + (cfg.comm_cost if comm_ok else 0.0)
    state.energy = clamp(state.energy + gen - cost, 0.0, cfg.energy_max)
    if state.energy <= 1e-6:
        state.blackout_count += 1

    # Data
    collected = cfg.data_collect_hi if state.energy >= 40.0 else cfg.data_collect_lo
    state.data_buffer += collected
    overflow = max(0.0, state.data_buffer - cfg.data_buffer_max)
    if overflow > 0:
        state.data_lost += overflow
        state.data_buffer = cfg.data_buffer_max

    down = 0.0
    if comm_ok:
        down = min(state.data_buffer, cfg.data_downlink_cap)
        state.data_buffer -= down
        state.data_downlinked += down

    # True vs estimate
    u_true = np.array([dβ[0] * state.p_true[0], dβ[1] * state.p_true[1]], dtype=float)
    u_est = np.array([dβ[0] * state.p_est[0], dβ[1] * state.p_est[1]], dtype=float)

    rcs_bias = rng.normal(0, cfg.rcs_sigma_per_sqrt_maneuver * math.sqrt(max(maneuvers, 0.0)), size=(2,))
    state.B_true = state.B_true + section.S @ u_true + rcs_bias
    state.B_est = state.B_est + section.S @ u_est

    # Observation
    B_obs = state.B_true + rng.normal(0, cfg.meas_sigma_km, size=(2,))
    state.B_obs_last = B_obs

    # OD update
    od_gain_eff = section.od_gain * (0.35 if state.energy < 30.0 else 1.0)
    state.p_est, state.P_cov = od_update_gains(B_obs, state.B_est, dβ, section, state, cfg, od_gain_eff)

    state.log.append(
        {
            "turn": int(state.k + 1),
            "section": section.name,
            "t_day": float(section.t_day),
            "comm_ok": int(comm_ok),
            "earth_angle_base_deg": float(earth_angle_base_deg(state, cfg, sections)),
            "earth_angle_deg": float(predicted_earth_angle_deg(state.beta_in, state.beta_out, state, cfg, sections)),
            "beta_in": float(state.beta_in),
            "beta_out": float(state.beta_out),
            "maneuvers_used": float(maneuvers),
            "maneuvers_left": float(state.maneuvers_left),
            "energy": float(state.energy),
            "data_downlinked": float(down),
            "data_buffer": float(state.data_buffer),
            "data_lost_total": float(state.data_lost),
            "dist_to_target_km": float(l2(state.B_true - cfg.target)),
        }
    )

    state.k += 1
    if state.k >= len(sections):
        state.phase = "result"

    state.rng_state["bitgen"] = rng.bit_generator.state


# -----------------------------
# Score (restored)
# -----------------------------
def score_game(state: GameState, cfg: GameConfig):
    dist = l2(state.B_true - cfg.target)
    used = cfg.maneuver_budget - state.maneuvers_left

    base = 10000.0
    dist_pen = 0.65 * dist
    manv_pen = 0.25 * used
    data_bonus = 55.0 * state.data_downlinked
    energy_bonus = 8.0 * state.energy
    data_loss_pen = 25.0 * state.data_lost
    blackout_pen = 600.0 * state.blackout_count

    s = base - dist_pen - manv_pen + data_bonus + energy_bonus - data_loss_pen - blackout_pen
    s = max(0.0, s)
    return s, {
        "final_distance_km": float(dist),
        "maneuvers_used": float(used),
        "energy_left": float(state.energy),
        "science_downlinked": float(state.data_downlinked),
        "data_lost": float(state.data_lost),
        "blackouts": int(state.blackout_count),
    }


# -----------------------------
# Plots (Matplotlib)
# -----------------------------
def controllability_poly(section: Section) -> np.ndarray:
    di, do = section.dbeta_in_max, section.dbeta_out_max
    S = section.S
    corners = []
    for si in (-di, di):
        for so in (-do, do):
            corners.append(S @ np.array([si, so], dtype=float))
    C = np.mean(np.stack(corners), axis=0)
    ang = [math.atan2((p - C)[1], (p - C)[0]) for p in corners]
    order = np.argsort(ang)
    poly = np.stack([corners[i] for i in order] + [corners[order[0]]], axis=0)
    return poly

def ellipse_params(center: np.ndarray, cov: np.ndarray, k_sigma: float = 1.0) -> Tuple[float, float, float]:
    w, V = np.linalg.eigh(cov + np.eye(2) * 1e-9)
    w = np.maximum(w, 1e-9)
    order = np.argsort(w)[::-1]
    w = w[order]
    V = V[:, order]
    a = k_sigma * math.sqrt(float(w[0]))
    b = k_sigma * math.sqrt(float(w[1]))
    width = 2.0 * a
    height = 2.0 * b
    ang = math.degrees(math.atan2(V[1, 0], V[0, 0]))
    return width, height, ang

def annotate_outlined(ax, x, y, text, dx=10, dy=10):
    t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                    color="white", fontsize=10, ha="left", va="bottom")
    t.set_path_effects([pe.Stroke(linewidth=3, foreground="black"), pe.Normal()])
    return t

def plot_bplane(state: GameState, cfg: GameConfig, sections: List[Section], show_truth: bool):
    sec = sections[min(state.k, len(sections) - 1)]
    pv = preview_next(state, cfg, sections, sec)
    B_pred = pv["B_pred"]
    cov_pred = pv["cov_pred"]

    tighten = (state.k + 1) >= cfg.target_tighten_section
    target_r = cfg.target_radius_late_km if tighten else cfg.target_radius_early_km

    poly = controllability_poly(sec) + state.B_est.reshape(1, 2)

    fig = plt.figure(figsize=(11.0, 5.6), dpi=150)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor("#0b0f16")
    ax.set_facecolor("#0b0f16")

    ax.add_patch(Polygon(poly, closed=True, facecolor="#00d1ff", edgecolor="none", alpha=0.08, zorder=1))
    ax.plot(poly[:, 0], poly[:, 1], linestyle="--", linewidth=2.0, color="#00d1ff", alpha=0.85, label="制御可能範囲（境界）", zorder=2)

    ax.add_patch(Circle((cfg.target[0], cfg.target[1]), target_r, fill=False, linewidth=2.6, edgecolor="#ffcc00", alpha=0.95, zorder=3))
    ax.plot([], [], color="#ffcc00", linewidth=2.6, label="ターゲット半径")

    w, h, ang = ellipse_params(B_pred, cov_pred, k_sigma=cfg.pred_ellipse_sigma)
    ax.add_patch(Ellipse((B_pred[0], B_pred[1]), width=w, height=h, angle=ang, fill=False,
                         edgecolor="white", linewidth=2.2, linestyle=":", alpha=0.95, zorder=4))
    ax.plot([], [], color="white", linestyle=":", linewidth=2.2, label="予測範囲（1σ）")

    ax.scatter([cfg.target[0]], [cfg.target[1]], s=70, color="#8aa2c8", zorder=5, label="ターゲット中心")
    ax.scatter([state.B_est[0]], [state.B_est[1]], s=90, color="#5dade2", marker="s", zorder=6, label="推定点E（いま）")
    ax.scatter([B_pred[0]], [B_pred[1]], s=90, color="white", marker="o", zorder=7, label="予測中心")
    if state.B_obs_last is not None:
        ax.scatter([state.B_obs_last[0]], [state.B_obs_last[1]], s=95, color="#aab7b8", marker="P", zorder=6, label="観測点（前ターン）")
    if show_truth:
        ax.scatter([state.B_true[0]], [state.B_true[1]], s=95, color="#ff6b6b", marker="^", zorder=7, label="真値（先生）")

    annotate_outlined(ax, cfg.target[0], cfg.target[1], "ターゲット中心", dx=8, dy=-18)
    annotate_outlined(ax, state.B_est[0], state.B_est[1], "推定E", dx=8, dy=8)
    annotate_outlined(ax, B_pred[0], B_pred[1], "予測中心", dx=8, dy=8)

    ax.set_title("B-plane（的当て）", color="white", fontsize=15, pad=10)
    ax.set_xlabel("BT [km]", color="white")
    ax.set_ylabel("BR [km]", color="white")
    ax.tick_params(colors="#cbd5e1")
    ax.grid(True, color="#334155", alpha=0.55, linewidth=0.7)

    span = 14000.0
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)

    leg = ax.legend(loc="upper right", frameon=True, fontsize=10)
    leg.get_frame().set_facecolor("#0b1220")
    leg.get_frame().set_alpha(0.85)
    for text in leg.get_texts():
        text.set_color("white")

    plt.tight_layout()
    return fig

def plot_orbits_2d_nominal(state: GameState, cfg: GameConfig, sections: List[Section]):
    eph = cfg.eph
    n = len(sections)

    sec_now = sections[min(state.k, n-1)]
    t_now = sec_now.t_day
    e = ephem_at_day(t_now, eph)
    earth = e["earth"]
    venus = e["venus"]

    # nominal spacecraft track by index
    pts_sc = []
    for i in range(0, min(state.k, n-1) + 1):
        pts_sc.append(sc_nominal_at_index(i, n, eph)["sc"])
    pts_sc = np.stack(pts_sc, axis=0)

    ths = np.linspace(0, 2*math.pi, 400)
    earth_orb = np.stack([eph.r_earth*np.cos(ths), eph.r_earth*np.sin(ths)], axis=1)
    venus_orb = np.stack([eph.r_venus*np.cos(ths), eph.r_venus*np.sin(ths)], axis=1)

    fig = plt.figure(figsize=(11.0, 4.2), dpi=150)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor("#0b0f16")
    ax.set_facecolor("#0b0f16")

    ax.plot(earth_orb[:,0], earth_orb[:,1], color="#3b82f6", alpha=0.55, linewidth=1.8, label="地球軌道")
    ax.plot(venus_orb[:,0], venus_orb[:,1], color="#22c55e", alpha=0.55, linewidth=1.8, label="金星軌道")

    ax.scatter([0],[0], s=140, color="#ffcc00", label="太陽", zorder=6)
    ax.scatter([earth[0]],[earth[1]], s=85, color="#3b82f6", label="地球（いま）", zorder=7)
    ax.scatter([venus[0]],[venus[1]], s=85, color="#22c55e", label="金星（いま）", zorder=7)

    ax.plot(pts_sc[:,0], pts_sc[:,1], color="white", linewidth=2.8, alpha=0.95, label="IKAROS（計画：ノミナル）", zorder=8)
    ax.scatter([pts_sc[-1,0]],[pts_sc[-1,1]], color="white", s=90, zorder=9)

    # meet check at arrival
    vend = ephem_at_day(eph.t_end_day, eph)["venus"]
    scend = sc_nominal_at_index(n-1, n, eph)["sc"]
    err = float(np.linalg.norm(vend - scend))

    t = ax.text(0.02, 0.98, f"到着一致チェック：|Venus - Nominal| ≈ {err:.3e} AU（0に近いほど一致）",
                transform=ax.transAxes, ha="left", va="top", color="white", fontsize=11)
    t.set_path_effects([pe.Stroke(linewidth=3, foreground="black"), pe.Normal()])

    base = earth_angle_base_deg(state, cfg, sections)
    t2 = ax.text(0.02, 0.90, f"地球角（幾何ベース） ≈ {base:+.1f}°",
                 transform=ax.transAxes, ha="left", va="top", color="white", fontsize=11)
    t2.set_path_effects([pe.Stroke(linewidth=3, foreground="black"), pe.Normal()])

    ax.set_aspect("equal", adjustable="box")
    ax.set_title("位置関係（2D軌道図：ノミナル）", color="white", fontsize=14, pad=10)
    ax.set_xlabel("x [AU]", color="white")
    ax.set_ylabel("y [AU]", color="white")
    ax.tick_params(colors="#cbd5e1")
    ax.grid(True, color="#334155", alpha=0.5, linewidth=0.7)

    lim = 1.25
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    leg = ax.legend(loc="lower left", frameon=True, fontsize=10)
    leg.get_frame().set_facecolor("#0b1220")
    leg.get_frame().set_alpha(0.85)
    for text in leg.get_texts():
        text.set_color("white")

    plt.tight_layout()
    return fig

def beta_map_grids(state: GameState, cfg: GameConfig, sections: List[Section], step: float = 2.0):
    bmin, bmax = -35.0, 35.0
    xs = np.arange(bmin, bmax + 1e-9, step)
    ys = np.arange(bmin, bmax + 1e-9, step)
    X, Y = np.meshgrid(xs, ys)

    net = np.zeros_like(X, dtype=float)
    down = np.zeros_like(X, dtype=float)
    comm_ok = np.zeros_like(X, dtype=float)

    for i in range(Y.shape[0]):
        for j in range(X.shape[1]):
            bi = float(X[i, j]); bo = float(Y[i, j])
            be = beta_eff(bi, bo)
            gen = cfg.gen_scale * max(0.0, cosd(be))
            ok = comm_available(bi, bo, state, cfg, sections)
            comm_ok[i, j] = 1.0 if ok else 0.0
            cost = cfg.base_load + (cfg.comm_cost if ok else 0.0)
            net[i, j] = gen - cost
            down[i, j] = cfg.data_downlink_cap if ok else 0.0

    return xs, ys, net, down, comm_ok

def plot_beta_maps(state: GameState, cfg: GameConfig, sections: List[Section]):
    xs, ys, net, down, comm_ok = beta_map_grids(state, cfg, sections, step=2.0)

    fig = plt.figure(figsize=(10.6, 4.8), dpi=150)
    fig.patch.set_facecolor("#0b0f16")

    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    for ax in (ax1, ax2):
        ax.set_facecolor("#0b0f16")
        ax.tick_params(colors="#cbd5e1")

    im1 = ax1.imshow(net, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]], aspect="equal")
    ax1.set_title("電力収支", color="white", fontsize=12)
    ax1.set_xlabel("βin [deg]", color="white")
    ax1.set_ylabel("βout [deg]", color="white")
    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cb1.ax.tick_params(colors="#cbd5e1")
    cb1.set_label("収支", color="white")

    im2 = ax2.imshow(down, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]], aspect="equal",
                     vmin=0, vmax=max(1.0, float(cfg.data_downlink_cap)))
    ax2.set_title("DL量（通信できるときだけ）", color="white", fontsize=12)
    ax2.set_xlabel("βin [deg]", color="white")
    ax2.set_ylabel("βout [deg]", color="white")
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.ax.tick_params(colors="#cbd5e1")
    cb2.set_label("DL", color="white")

    for ax in (ax1, ax2):
        ax.contour(xs, ys, comm_ok, levels=[0.5], colors=["#9cff57"], linewidths=2.2)
        ax.contourf(xs, ys, comm_ok, levels=[-0.1, 0.5, 1.1], colors=["#00000000", "#9cff57"], alpha=0.12)
        ax.scatter([state.beta_in], [state.beta_out], s=80, color="white", edgecolor="black", linewidth=1.2, zorder=6)
        ax.text(state.beta_in + 1.2, state.beta_out + 1.2, "いま", color="white", fontsize=9,
                path_effects=[pe.Stroke(linewidth=3, foreground="black"), pe.Normal()])

    fig.text(0.5, 0.01, "緑の境界/面 = 通信OK領域（地球角±窓、電力が十分な場合）", ha="center", color="white", fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    return fig


# -----------------------------
# Plot: Geometry 3D (Plotly)
# -----------------------------
def rot_z(deg: float) -> np.ndarray:
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=float)

def rot_y(deg: float) -> np.ndarray:
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], dtype=float)

def geometry_3d_figure(state: GameState, cfg: GameConfig, sections: List[Section]) -> go.Figure:
    ea = predicted_earth_angle_deg(state.beta_in, state.beta_out, state, cfg, sections)
    be = beta_eff(state.beta_in, state.beta_out)
    bp = beta_pointing(state.beta_in, state.beta_out)

    sun = np.array([1.0, 0.0, 0.0])
    earth = np.array([math.cos(math.radians(ea)), math.sin(math.radians(ea)), 0.0])
    n = rot_z(bp) @ rot_y(be) @ np.array([1.0, 0.0, 0.0])

    n_norm = n / (np.linalg.norm(n) + 1e-9)
    tmp = np.array([0.0, 0.0, 1.0])
    if abs(float(n_norm.dot(tmp))) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    u = np.cross(n_norm, tmp); u = u / (np.linalg.norm(u) + 1e-9)
    v = np.cross(n_norm, u); v = v / (np.linalg.norm(v) + 1e-9)
    s = 0.65
    corners = np.stack([ s*u + s*v,
                         s*u - s*v,
                        -s*u - s*v,
                        -s*u + s*v,
                         s*u + s*v ], axis=0)

    def vec_trace(vec, name, color):
        return go.Scatter3d(x=[0, vec[0]], y=[0, vec[1]], z=[0, vec[2]], mode="lines",
                            line=dict(width=6, color=color), name=name)

    fig = go.Figure()
    fig.add_trace(vec_trace(sun, "太陽光", "#ffcc00"))
    fig.add_trace(vec_trace(earth, "地球方向", "#9cff57"))
    fig.add_trace(vec_trace(n_norm, "帆法線", "#00d1ff"))
    fig.add_trace(go.Scatter3d(x=corners[:,0], y=corners[:,1], z=corners[:,2], mode="lines",
                               line=dict(width=5, color="white"), name="帆（平面）"))
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers+text",
                               marker=dict(size=6, color="white"),
                               text=["IKAROS"], textposition="bottom center", name="IKAROS"))
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        title=f"幾何（3D概念図）  βeff={be:.1f}°, βpoint={bp:+.1f}°, 地球角={ea:+.1f}°",
        scene=dict(
            xaxis=dict(title="X", showbackground=False, color="white"),
            yaxis=dict(title="Y", showbackground=False, color="white"),
            zaxis=dict(title="Z", showbackground=False, color="white"),
            bgcolor="rgba(0,0,0,0)"
        ),
        legend=dict(font=dict(color="white")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="IKAROS B-plane Darts v11", layout="wide")
st.title("🎯 IKAROS：B-plane ダーツ（運用ゲーム）")
st.caption("v11：japanize_matplotlib依存を削除（Python 3.13対応）。B-planeメイン、2D軌道図はノミナルが金星と一致。")

sections = build_sections()
cfg = GameConfig()

with st.sidebar:
    st.header("設定")
    seed = st.number_input("シード（同じ問題を再現）", min_value=1, max_value=999999, value=42, step=1)
    show_truth = st.toggle("先生モード：真値を表示", value=False)
    st.divider()
    st.subheader("日本語フォント")
    if JP_FONT_NAME:
        st.caption(f"使用フォント：{JP_FONT_NAME}")
    else:
        st.warning("日本語フォントが見つからず、文字が□になる可能性があります。")

    st.divider()
    st.subheader("学習ポイント")
    st.markdown(
        """
- **SRPは弱い** → “調整” しかできない  
- **投入誤差**がある → 放置は負け筋  
- **通信ウィンドウ**は軌道幾何で決まる → βで“指向”を合わせる  
- でも βを増やすと **発電が落ちる**  
"""
    )

seed_int = int(seed)
if "bplane_state_v11" not in st.session_state or st.session_state.get("bplane_seed_v11") != seed_int:
    st.session_state.bplane_state_v11 = init_game(cfg, sections, seed=seed_int)
    st.session_state.bplane_seed_v11 = seed_int
    st.session_state.page_v11 = "Play"

state: GameState = st.session_state.bplane_state_v11

def rerun():
    (st.rerun() if hasattr(st, "rerun") else st.experimental_rerun())

def reset():
    st.session_state.bplane_state_v11 = init_game(cfg, sections, seed=seed_int)
    st.session_state.page_v11 = "Play"
    rerun()

if state.phase == "result":
    st.session_state.page_v11 = "Result"

page = st.radio("ページ", ["Play", "Result"], horizontal=True, index=(0 if st.session_state.page_v11 == "Play" else 1))
st.session_state.page_v11 = page


def render_play():
    sec = sections[min(state.k, len(sections) - 1)]
    ea_base = earth_angle_base_deg(state, cfg, sections)
    ea = predicted_earth_angle_deg(state.beta_in, state.beta_out, state, cfg, sections)
    comm_ok = comm_available(state.beta_in, state.beta_out, state, cfg, sections)

    st.progress(min(1.0, state.k / len(sections)))
    st.write(f"進捗：**{state.k}/{len(sections)}** セクション完了（全{len(sections)}）  |  現在：**{sec.name}**（t≈{sec.t_day:.0f}日）")

    a1, a2, a3, a4, a5 = st.columns([1.0, 1.1, 1.1, 1.3, 1.5])
    with a1:
        st.metric("通信", fmt_bool(comm_ok))
    with a2:
        st.metric("バッテリ", f"{state.energy:.0f}/{cfg.energy_max:.0f}")
    with a3:
        st.metric("地球角(幾何)", f"{ea_base:+.1f}°")
    with a4:
        st.metric("地球角(指向後)", f"{ea:+.1f}°")
    with a5:
        btn_next = st.button("▶ このセクションを実行（進める）", use_container_width=True, disabled=(state.phase == "result"))
        btn_reset = st.button("🔁 リセット", use_container_width=True)
    if btn_reset:
        reset()
    if btn_next:
        execute_section(state, cfg, sections)
        rerun()

    st.subheader("B-plane（メイン）")
    st.pyplot(plot_bplane(state, cfg, sections, show_truth=show_truth), use_container_width=True)

    if comm_ok:
        st.success("このβなら通信OK見込み（コマンド送信＆データ下ろし）。")
    else:
        st.warning("このβだと通信NG見込み → 実行するとΔβ=0固定＆DLできない。")

    left, right = st.columns([1.0, 1.0], gap="large")

    with left:
        st.subheader("位置関係（2D軌道図：ノミナル）")
        st.pyplot(plot_orbits_2d_nominal(state, cfg, sections), use_container_width=True)

        if state.log:
            df = pd.DataFrame(state.log)
            st.subheader("ライブ推移")
            st.line_chart(df.set_index("turn")[["dist_to_target_km"]], height=170)
            st.line_chart(df.set_index("turn")[["energy", "earth_angle_deg"]], height=200)

    with right:
        st.subheader("βin×βout マップ（幾何 + 指向 + 電力）")
        st.pyplot(plot_beta_maps(state, cfg, sections), use_container_width=True)

        st.subheader("幾何（3D表示）")
        st.caption("ドラッグで回転できます。")
        st.plotly_chart(geometry_3d_figure(state, cfg, sections), use_container_width=True)

        st.subheader("コマンド（βin / βout）")
        if not sec.uplink_possible:
            st.error("このセクションは NO-LINK：通信不可（コマンド固定）。")

        cA, cB = st.columns(2)
        with cA:
            bi = st.slider("βin [deg]", -35.0, 35.0, float(state.beta_in), 1.0)
        with cB:
            bo = st.slider("βout [deg]", -35.0, 35.0, float(state.beta_out), 1.0)

        state.beta_in = float(bi)
        state.beta_out = float(bo)


def render_result():
    st.header("📊 リザルト")
    score, bd = score_game(state, cfg)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("スコア", f"{score:.0f}")
    c2.metric("最終距離（B-plane）", f"{bd['final_distance_km']:.0f} km")
    c3.metric("データ下ろし", f"{bd['science_downlinked']:.0f}")
    c4.metric("電力残", f"{bd['energy_left']:.0f}")

    st.subheader("B-plane（最終）")
    st.pyplot(plot_bplane(state, cfg, sections, show_truth=True), use_container_width=True)

    st.subheader("位置関係（2D軌道図：ノミナル）")
    st.pyplot(plot_orbits_2d_nominal(state, cfg, sections), use_container_width=True)

    if state.log:
        df = pd.DataFrame(state.log)
        st.subheader("推移まとめ")
        st.line_chart(df.set_index("turn")[["dist_to_target_km", "energy", "earth_angle_deg", "data_buffer", "data_lost_total"]], height=280)

    if st.button("🔁 もう一回（リセット）", use_container_width=True):
        reset()


if page == "Play":
    render_play()
else:
    render_result()

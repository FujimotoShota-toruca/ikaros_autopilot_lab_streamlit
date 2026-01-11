# IKAROS：B-plane ダーツ（適応誘導オペレーション）
# Streamlit + Vega-Lite (direct spec)
#
# v7 improvements (UI + clarity):
# - Fix β map visibility: remove "black mask" and instead overlay COMM-OK points + clearer color scales
# - Add B-plane "予測点（次のターンに進めた場合）"
# - Make target ring / controllability region more distinguishable (colors + fill)
# - More visual "Result" page with charts; basic page navigation (Play / Result)
# - Put "進める" button near top (easy access)
# - Live charts: avoid "almost no change" by disabling zero-baseline and using auto domain
#
# Note: educational abstraction, not real flight dynamics.
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import streamlit as st


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def l2(xy: np.ndarray) -> float:
    return float(np.linalg.norm(xy))


def cosd(deg: float) -> float:
    return math.cos(math.radians(deg))


def fmt_bool(b: bool) -> str:
    return "🟢OK" if b else "🔴NG"


@dataclass
class Section:
    name: str
    S: np.ndarray
    dbeta_in_max: float
    dbeta_out_max: float
    uplink_possible: bool
    maneuvers_per_deg: float
    od_gain: float
    earth_angle_bias_deg: float


def build_sections() -> List[Section]:
    def mat(a, b, c, d):
        return np.array([[a, b], [c, d]], dtype=float)

    S_pre = mat(180, 40, -20, 140)
    S_pre2 = mat(210, 60, -40, 170)
    S_pre3 = mat(240, 70, -60, 190)

    S_post = mat(520, 130, -90, 430)
    S_post2 = mat(560, 150, -110, 460)
    S_post3 = mat(600, 170, -120, 500)
    S_post4 = mat(640, 190, -140, 520)

    return [
        Section("Section 1", S_pre, 6, 6, True, 65, 0.45, earth_angle_bias_deg=+5),
        Section("Section 2", S_pre2, 6, 6, True, 80, 0.50, earth_angle_bias_deg=+12),
        Section("Section 3", S_pre3, 5, 5, True, 95, 0.55, earth_angle_bias_deg=+25),
        Section("Section 4 (NO-LINK)", S_post, 0, 0, False, 0, 0.60, earth_angle_bias_deg=+35),
        Section("Section 5", S_post2, 18, 18, True, 45, 0.70, earth_angle_bias_deg=+18),
        Section("Section 6", S_post3, 18, 18, True, 35, 0.78, earth_angle_bias_deg=+8),
        Section("Section 7", S_post4, 15, 15, True, 30, 0.85, earth_angle_bias_deg=+2),
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

    comm_window_deg: float = 20.0
    energy_max: float = 200.0
    energy_init: float = 140.0
    energy_min_for_comm: float = 30.0
    base_load: float = 70.0
    gen_scale: float = 90.0
    comm_cost: float = 10.0
    maneuver_energy_scale: float = 0.02

    data_buffer_max: float = 60.0
    data_collect_hi: float = 12.0
    data_collect_lo: float = 4.0
    data_downlink_cap: float = 18.0

    beta_to_earth_coupling: float = 0.7


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

    p_true = np.array(
        [1.0 + rng.normal(0, cfg.sigma_gain_in0), 1.0 + rng.normal(0, cfg.sigma_gain_out0)],
        dtype=float,
    )
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


def beta_eff(bin_deg: float, bout_deg: float) -> float:
    return 0.5 * (abs(bin_deg) + abs(bout_deg))


def predicted_earth_angle_deg(bin_deg: float, bout_deg: float, section: Section, cfg: GameConfig) -> float:
    return float(section.earth_angle_bias_deg + cfg.beta_to_earth_coupling * beta_eff(bin_deg, bout_deg))


def comm_available(bin_deg: float, bout_deg: float, section: Section, cfg: GameConfig, energy: float) -> bool:
    if not section.uplink_possible:
        return False
    ea = predicted_earth_angle_deg(bin_deg, bout_deg, section, cfg)
    return bool((abs(ea) <= cfg.comm_window_deg) and (energy >= cfg.energy_min_for_comm))


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


def clamp_dbeta(dβ: np.ndarray, section: Section) -> np.ndarray:
    d = dβ.copy()
    d[0] = clamp(d[0], -section.dbeta_in_max, section.dbeta_in_max)
    d[1] = clamp(d[1], -section.dbeta_out_max, section.dbeta_out_max)
    return d


def preview_next_est_point(state: GameState, cfg: GameConfig, section: Section) -> Tuple[np.ndarray, np.ndarray, bool]:
    plan = np.array([cfg.plan_beta_in_deg, cfg.plan_beta_out_deg], dtype=float)
    cmd = np.array([state.beta_in, state.beta_out], dtype=float)
    dβ = cmd - plan

    comm_ok = comm_available(float(cmd[0]), float(cmd[1]), section, cfg, state.energy)
    if not comm_ok:
        dβ = np.array([0.0, 0.0], dtype=float)
    dβ = clamp_dbeta(dβ, section)

    u_est = np.array([dβ[0] * state.p_est[0], dβ[1] * state.p_est[1]], dtype=float)
    B_pred = state.B_est + section.S @ u_est
    return B_pred, dβ, comm_ok


def execute_section(state: GameState, cfg: GameConfig, sections: List[Section]) -> None:
    rng = np.random.default_rng()
    rng.bit_generator.state = state.rng_state["bitgen"]

    section = sections[state.k]
    B_pred, dβ, comm_ok = preview_next_est_point(state, cfg, section)

    total_deg = abs(float(dβ[0])) + abs(float(dβ[1]))
    maneuvers = section.maneuvers_per_deg * total_deg
    if maneuvers > state.maneuvers_left:
        scale = 0.0 if state.maneuvers_left <= 0 else (state.maneuvers_left / max(maneuvers, 1e-9))
        dβ = dβ * scale
        B_pred, _, comm_ok = preview_next_est_point(state, cfg, section)
        maneuvers = section.maneuvers_per_deg * (abs(float(dβ[0])) + abs(float(dβ[1])))

    state.maneuvers_left -= maneuvers

    cmd = np.array([state.beta_in, state.beta_out], dtype=float)
    be = beta_eff(float(cmd[0]), float(cmd[1])) if comm_ok else 0.0
    gen = cfg.gen_scale * max(0.0, cosd(be))
    cost = cfg.base_load + cfg.maneuver_energy_scale * maneuvers + (cfg.comm_cost if comm_ok else 0.0)
    state.energy = clamp(state.energy + gen - cost, 0.0, cfg.energy_max)
    if state.energy <= 1e-6:
        state.blackout_count += 1

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

    u_true = np.array([dβ[0] * state.p_true[0], dβ[1] * state.p_true[1]], dtype=float)
    u_est = np.array([dβ[0] * state.p_est[0], dβ[1] * state.p_est[1]], dtype=float)

    rcs_bias = rng.normal(0, cfg.rcs_sigma_per_sqrt_maneuver * math.sqrt(max(maneuvers, 0.0)), size=(2,))
    state.B_true = state.B_true + section.S @ u_true + rcs_bias
    state.B_est = state.B_est + section.S @ u_est

    B_obs = state.B_true + rng.normal(0, cfg.meas_sigma_km, size=(2,))
    state.B_obs_last = B_obs

    od_gain_eff = section.od_gain * (0.35 if state.energy < 30.0 else 1.0)
    state.p_est, state.P_cov = od_update_gains(B_obs, state.B_est, dβ, section, state, cfg, od_gain_eff)

    sigma = np.sqrt(np.diag(state.P_cov))
    dist = l2(state.B_true - cfg.target)
    ea = predicted_earth_angle_deg(float(cmd[0]), float(cmd[1]), section, cfg)

    state.log.append(
        {
            "turn": int(state.k + 1),
            "section": section.name,
            "comm_ok": int(comm_ok),
            "earth_angle_deg": float(ea),
            "beta_in": float(cmd[0]),
            "beta_out": float(cmd[1]),
            "beta_eff_deg": float(be),
            "applied_dbeta_in": float(dβ[0]),
            "applied_dbeta_out": float(dβ[1]),
            "maneuvers_used": float(maneuvers),
            "maneuvers_left": float(state.maneuvers_left),
            "energy": float(state.energy),
            "data_downlinked": float(down),
            "data_buffer": float(state.data_buffer),
            "data_lost_total": float(state.data_lost),
            "BT_true_km": float(state.B_true[0]),
            "BR_true_km": float(state.B_true[1]),
            "BT_est_km": float(state.B_est[0]),
            "BR_est_km": float(state.B_est[1]),
            "dist_to_target_km": float(dist),
            "gain_in_est": float(state.p_est[0]),
            "gain_out_est": float(state.p_est[1]),
            "sigma_gain_in": float(sigma[0]),
            "sigma_gain_out": float(sigma[1]),
        }
    )

    state.k += 1
    if state.k >= len(sections):
        state.phase = "result"

    state.rng_state["bitgen"] = rng.bit_generator.state


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

    breakdown = {
        "base": base,
        "距離ペナルティ": -dist_pen,
        "マヌーバペナルティ": -manv_pen,
        "データボーナス": data_bonus,
        "電力ボーナス": energy_bonus,
        "データ損失ペナルティ": -data_loss_pen,
        "ブラックアウトペナルティ": -blackout_pen,
        "score": s,
        "final_distance_km": float(dist),
        "maneuvers_used": float(used),
        "energy_left": float(state.energy),
        "science_downlinked": float(state.data_downlinked),
        "data_lost": float(state.data_lost),
        "blackouts": int(state.blackout_count),
    }
    return s, breakdown


def compute_controllability_polygon(section: Section) -> np.ndarray:
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


def vega_timeseries_spec(log: List[Dict], y_field: str, y_title: str, height: int = 160) -> Dict:
    vals = [{"turn": int(r.get("turn", i + 1)), "section": r.get("section", f"{i+1}"), "y": float(r.get(y_field, 0.0))} for i, r in enumerate(log)]
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"values": vals},
        "height": height,
        "mark": {"type": "line", "point": True},
        "encoding": {
            "x": {"field": "turn", "type": "quantitative", "title": "ターン", "tickMinStep": 1},
            "y": {"field": "y", "type": "quantitative", "title": y_title, "scale": {"zero": False}},
            "tooltip": [{"field": "section", "type": "nominal"}, {"field": "y", "type": "quantitative"}],
        },
        "config": {"view": {"stroke": None}},
    }


def vega_breakdown_bar(bd: Dict) -> Dict:
    keys = ["距離ペナルティ", "マヌーバペナルティ", "データボーナス", "電力ボーナス", "データ損失ペナルティ", "ブラックアウトペナルティ"]
    vals = [{"k": k, "v": float(bd.get(k, 0.0))} for k in keys]
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"values": vals},
        "height": 220,
        "mark": {"type": "bar"},
        "encoding": {
            "y": {"field": "k", "type": "nominal", "title": ""},
            "x": {"field": "v", "type": "quantitative", "title": "加点/減点（＋が良い）"},
            "tooltip": [{"field": "k", "type": "nominal"}, {"field": "v", "type": "quantitative", "format": ".0f"}],
        },
        "config": {"view": {"stroke": None}},
    }


def vega_bplane_spec(state: GameState, cfg: GameConfig, sections: List[Section], show_truth: bool) -> Dict:
    section = sections[min(state.k, len(sections) - 1)]
    B_pred, _, comm_ok = preview_next_est_point(state, cfg, section)

    poly = compute_controllability_polygon(section) + state.B_est.reshape(1, 2)
    poly_vals = [{"BT": float(p[0]), "BR": float(p[1]), "idx": i} for i, p in enumerate(poly)]

    tighten = (state.k + 1) >= cfg.target_tighten_section
    target_r = cfg.target_radius_late_km if tighten else cfg.target_radius_early_km

    ring_vals = []
    for i in range(65):
        th = 2 * math.pi * i / 64
        ring_vals.append({"BT": float(cfg.target[0] + target_r * math.cos(th)),
                          "BR": float(cfg.target[1] + target_r * math.sin(th)),
                          "i": i})

    pts = [
        {"BT": float(cfg.target[0]), "BR": float(cfg.target[1]), "kind": "ターゲット中心"},
        {"BT": float(state.B_est[0]), "BR": float(state.B_est[1]), "kind": "推定点E（いま）"},
        {"BT": float(B_pred[0]), "BR": float(B_pred[1]), "kind": "予測点（次へ）" if comm_ok else "予測点（次へ/固定）"},
    ]
    if show_truth:
        pts.append({"BT": float(state.B_true[0]), "BR": float(state.B_true[1]), "kind": "真値（先生）"})
    if state.B_obs_last is not None:
        pts.append({"BT": float(state.B_obs_last[0]), "BR": float(state.B_obs_last[1]), "kind": "観測点（前ターン）"})

    all_bt = [p["BT"] for p in pts] + [p["BT"] for p in poly_vals] + [p["BT"] for p in ring_vals]
    all_br = [p["BR"] for p in pts] + [p["BR"] for p in poly_vals] + [p["BR"] for p in ring_vals]
    span = max(12000.0, max(map(abs, all_bt + [0])), max(map(abs, all_br + [0])))
    span = float(span * 1.15)

    ring_color = "#ffcc00"
    poly_color = "#00d1ff"
    poly_fill = "#00d1ff"

    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "height": 440,
        "encoding": {
            "x": {"field": "BT", "type": "quantitative", "title": "BT [km]", "scale": {"domain": [-span, span]}},
            "y": {"field": "BR", "type": "quantitative", "title": "BR [km]", "scale": {"domain": [-span, span]}},
        },
        "layer": [
            {"data": {"values": ring_vals},
             "mark": {"type": "line", "stroke": ring_color, "strokeWidth": 2, "opacity": 0.55},
             "encoding": {"order": {"field": "i", "type": "quantitative"}}},
            {"data": {"values": poly_vals},
             "mark": {"type": "area", "fill": poly_fill, "opacity": 0.08},
             "encoding": {"order": {"field": "idx", "type": "quantitative"}}},
            {"data": {"values": poly_vals},
             "mark": {"type": "line", "stroke": poly_color, "strokeWidth": 2, "opacity": 0.55},
             "encoding": {"order": {"field": "idx", "type": "quantitative"}}},
            {"data": {"values": pts},
             "mark": {"type": "point", "filled": True, "size": 120},
             "encoding": {
                 "shape": {"field": "kind", "type": "nominal", "legend": {"title": ""}},
                 "tooltip": [
                     {"field": "kind", "type": "nominal"},
                     {"field": "BT", "type": "quantitative", "format": ".0f"},
                     {"field": "BR", "type": "quantitative", "format": ".0f"},
                 ],
             }},
            {"data": {"values": pts},
             "mark": {"type": "text", "align": "left", "dx": 8, "dy": -8},
             "encoding": {"text": {"field": "kind", "type": "nominal"}}},
        ],
        "config": {"axis": {"labelFontSize": 12, "titleFontSize": 12}, "view": {"stroke": None}},
    }


def build_beta_map_data(section: Section, cfg: GameConfig, energy: float, step: float = 2.0) -> List[Dict]:
    vals: List[Dict] = []
    bmin, bmax = -35.0, 35.0
    b = bmin
    while b <= bmax + 1e-9:
        bo = bmin
        while bo <= bmax + 1e-9:
            be = beta_eff(b, bo)
            gen = cfg.gen_scale * max(0.0, cosd(be))
            ea = predicted_earth_angle_deg(b, bo, section, cfg)
            comm_ok = int(comm_available(b, bo, section, cfg, energy))
            cost = cfg.base_load + (cfg.comm_cost if comm_ok else 0.0)
            net = gen - cost
            down = cfg.data_downlink_cap if comm_ok else 0.0
            vals.append({
                "beta_in": float(b),
                "beta_out": float(bo),
                "beta_eff": float(be),
                "gen": float(gen),
                "cost": float(cost),
                "net": float(net),
                "earth_angle": float(ea),
                "comm_ok": int(comm_ok),
                "downlink": float(down),
            })
            bo += step
        b += step
    return vals


def vega_beta_map_spec(vals: List[Dict], title: str, color_field: str, color_title: str,
                       point: Tuple[float, float], scheme: str, diverging: bool = False) -> Dict:
    domain = None
    if diverging:
        mx = max(abs(float(v[color_field])) for v in vals) if vals else 1.0
        mx = max(mx, 1.0)
        domain = [-mx, mx]

    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": {"text": title, "fontSize": 14},
        "width": 360,
        "height": 360,
        "layer": [
            {"data": {"values": vals},
             "mark": {"type": "rect"},
             "encoding": {
                 "x": {"field": "beta_in", "type": "quantitative", "title": "βin [deg]"},
                 "y": {"field": "beta_out", "type": "quantitative", "title": "βout [deg]"},
                 "color": {"field": color_field, "type": "quantitative", "title": color_title,
                           "scale": {"scheme": scheme, **({"domain": domain} if domain else {})}},
                 "tooltip": [
                     {"field": "beta_in", "type": "quantitative", "format": ".1f"},
                     {"field": "beta_out", "type": "quantitative", "format": ".1f"},
                     {"field": "beta_eff", "type": "quantitative", "format": ".1f", "title": "βeff"},
                     {"field": "gen", "type": "quantitative", "format": ".0f", "title": "発電"},
                     {"field": "cost", "type": "quantitative", "format": ".0f", "title": "消費"},
                     {"field": "net", "type": "quantitative", "format": "+.0f", "title": "電力収支"},
                     {"field": "earth_angle", "type": "quantitative", "format": ".1f", "title": "地球角"},
                     {"field": "comm_ok", "type": "quantitative", "title": "通信OK(1/0)"},
                     {"field": "downlink", "type": "quantitative", "title": "DL量"},
                 ],
             }},
            {"data": {"values": vals},
             "transform": [{"filter": "datum.comm_ok == 1"}],
             "mark": {"type": "point", "shape": "square", "filled": False, "size": 55, "strokeWidth": 1.5, "opacity": 0.9},
             "encoding": {
                 "x": {"field": "beta_in", "type": "quantitative"},
                 "y": {"field": "beta_out", "type": "quantitative"},
                 "color": {"value": "#9cff57"},
             }},
            {"data": {"values": [{"beta_in": float(point[0]), "beta_out": float(point[1])}]},
             "mark": {"type": "point", "filled": True, "size": 140},
             "encoding": {
                 "x": {"field": "beta_in", "type": "quantitative"},
                 "y": {"field": "beta_out", "type": "quantitative"},
                 "color": {"value": "white"},
             }},
        ],
        "config": {"view": {"stroke": None}},
    }


# -----------------------------
# UI + Pages
# -----------------------------
st.set_page_config(page_title="IKAROS B-plane Darts", layout="wide")
st.title("🎯 IKAROS：B-plane ダーツ（適応誘導オペレーション）")
st.caption("B-planeの的当てを、電力・通信・データで“運用ゲーム”にした教材。")

sections = build_sections()
cfg = GameConfig()

with st.sidebar:
    st.header("設定")
    seed = st.number_input("シード（同じ問題を再現）", min_value=1, max_value=999999, value=42, step=1)
    show_truth = st.toggle("先生モード：真値を表示", value=False)
    st.divider()
    st.subheader("学習ポイント（短く）")
    st.markdown(
        """
- **SRPは弱い** → 一発で大きく動かない  
- **投入誤差**がある → 放置は負け筋  
- **通信ウィンドウ**と**電力**が、操作そのものを縛る  
"""
    )

seed_int = int(seed)

if "bplane_state_v7" not in st.session_state or st.session_state.get("bplane_seed_v7") != seed_int:
    st.session_state.bplane_state_v7 = init_game(cfg, sections, seed=seed_int)
    st.session_state.bplane_seed_v7 = seed_int
    st.session_state.page_v7 = "Play"

state: GameState = st.session_state.bplane_state_v7


def rerun():
    (st.rerun() if hasattr(st, "rerun") else st.experimental_rerun())


def reset():
    st.session_state.bplane_state_v7 = init_game(cfg, sections, seed=seed_int)
    st.session_state.page_v7 = "Play"
    rerun()


if state.phase == "result":
    st.session_state.page_v7 = "Result"

page = st.radio("ページ", ["Play", "Result"], horizontal=True, index=(0 if st.session_state.page_v7 == "Play" else 1))
st.session_state.page_v7 = page


def render_play():
    sec = sections[min(state.k, len(sections) - 1)]
    comm_preview = comm_available(state.beta_in, state.beta_out, sec, cfg, state.energy)
    ea_preview = predicted_earth_angle_deg(state.beta_in, state.beta_out, sec, cfg)
    be = beta_eff(state.beta_in, state.beta_out)

    st.progress(min(1.0, state.k / len(sections)))
    st.write(f"進捗：**{state.k}/{len(sections)}** セクション完了（全7）  |  現在：**{sec.name}**")

    a1, a2, a3, a4 = st.columns([1.2, 1.2, 1.2, 1.4])
    with a1:
        st.metric("通信", fmt_bool(comm_preview))
    with a2:
        st.metric("バッテリ", f"{state.energy:.0f}/{cfg.energy_max:.0f}")
    with a3:
        st.metric("地球角", f"{ea_preview:.1f}°")
    with a4:
        btn_next = st.button("▶ このセクションを実行（進める）", use_container_width=True, disabled=(state.phase == "result"))
        btn_reset = st.button("🔁 リセット", use_container_width=True)

    if btn_reset:
        reset()
    if btn_next:
        execute_section(state, cfg, sections)
        rerun()

    left, right = st.columns([1.55, 1.05], gap="large")

    with left:
        st.subheader("B-plane（的当て）")
        st.caption("黄色=ターゲット半径 / 水色=このセクションで動かせる範囲 / 白=予測点（次へ）")
        st.vega_lite_chart(vega_bplane_spec(state, cfg, sections, show_truth), use_container_width=True)

        if comm_preview:
            st.success("このβだと通信できそう（コマンド送信&データ下ろし）。")
        else:
            st.warning("このβだと通信できない見込み → 実行するとΔβ=0固定＆データ下ろせない。")

        if state.log:
            st.markdown("### ライブ推移")
            g1, g2 = st.columns(2)
            with g1:
                st.vega_lite_chart(vega_timeseries_spec(state.log, "dist_to_target_km", "ターゲット距離 [km]"), use_container_width=True)
            with g2:
                st.vega_lite_chart(vega_timeseries_spec(state.log, "energy", "バッテリ"), use_container_width=True)

            g3, g4 = st.columns(2)
            with g3:
                st.vega_lite_chart(vega_timeseries_spec(state.log, "earth_angle_deg", "地球角 [deg]"), use_container_width=True)
            with g4:
                st.vega_lite_chart(vega_timeseries_spec(state.log, "data_buffer", "データバッファ"), use_container_width=True)

    with right:
        st.subheader("βin×βout マップ")
        st.caption("色=電力収支 / DL量。緑の四角=通信OK領域。白丸=現在の入力。")

        beta_vals = build_beta_map_data(sec, cfg, energy=state.energy, step=2.0)
        m1, m2 = st.columns(2)
        with m1:
            st.vega_lite_chart(
                vega_beta_map_spec(beta_vals, "電力収支", "net", "電力収支", (state.beta_in, state.beta_out), scheme="redblue", diverging=True),
                use_container_width=True,
            )
        with m2:
            st.vega_lite_chart(
                vega_beta_map_spec(beta_vals, "DL量（通信できるときだけ）", "downlink", "DL量", (state.beta_in, state.beta_out), scheme="blues", diverging=False),
                use_container_width=True,
            )

        st.subheader("コマンド（βin / βout）")
        if state.phase != "result" and not sec.uplink_possible:
            st.error("このセクションは NO-LINK：通信不可（コマンド固定）。")

        cA, cB = st.columns(2)
        with cA:
            bi = st.slider("βin [deg]", -35.0, 35.0, float(state.beta_in), 1.0, disabled=(state.phase == "result"))
            bi = float(st.number_input("βin 直打ち", -90.0, 90.0, bi, 1.0, disabled=(state.phase == "result")))
        with cB:
            bo = st.slider("βout [deg]", -35.0, 35.0, float(state.beta_out), 1.0, disabled=(state.phase == "result"))
            bo = float(st.number_input("βout 直打ち", -90.0, 90.0, bo, 1.0, disabled=(state.phase == "result")))

        state.beta_in = bi
        state.beta_out = bo

        comm_preview2 = comm_available(bi, bo, sec, cfg, state.energy)
        ea2 = predicted_earth_angle_deg(bi, bo, sec, cfg)
        be2 = beta_eff(bi, bo)
        gen2 = cfg.gen_scale * max(0.0, cosd(be2))
        cost2 = cfg.base_load + (cfg.comm_cost if comm_preview2 else 0.0)
        net2 = gen2 - cost2
        tighten = (state.k + 1) >= cfg.target_tighten_section
        sigma = np.sqrt(np.diag(state.P_cov))

        st.subheader("運用ステータス（このβだと…）")
        s1, s2 = st.columns(2)
        s1.metric("通信", fmt_bool(comm_preview2))
        s2.metric("βeff", f"{be2:.1f}°")
        s3, s4 = st.columns(2)
        s3.metric("地球角", f"{ea2:.1f}°")
        s4.metric("電力収支", f"{net2:+.0f}")
        st.caption(f"ターゲット半径：{(cfg.target_radius_late_km if tighten else cfg.target_radius_early_km):.0f} km  |  残り予算：{state.maneuvers_left:.0f}")
        st.caption(f"推定ゲイン：in={state.p_est[0]:.2f}±{sigma[0]:.2f}, out={state.p_est[1]:.2f}±{sigma[1]:.2f}")

    st.subheader("ログ（必要なら）")
    if state.log:
        st.dataframe(state.log, use_container_width=True, hide_index=True)
    else:
        st.caption("まだ実行していません。")


def render_result():
    st.header("📊 リザルト")
    score, bd = score_game(state, cfg)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("スコア", f"{score:.0f}")
    c2.metric("最終距離", f"{bd['final_distance_km']:.0f} km")
    c3.metric("データ下ろし", f"{bd['science_downlinked']:.0f}")
    c4.metric("電力残", f"{bd['energy_left']:.0f}")

    st.subheader("B-plane（最終）")
    st.vega_lite_chart(vega_bplane_spec(state, cfg, sections, show_truth=True), use_container_width=True)

    t1, t2 = st.columns([1.2, 1.0], gap="large")
    with t1:
        st.subheader("推移まとめ")
        if state.log:
            st.vega_lite_chart(vega_timeseries_spec(state.log, "dist_to_target_km", "ターゲット距離 [km]", height=180), use_container_width=True)
            g1, g2 = st.columns(2)
            with g1:
                st.vega_lite_chart(vega_timeseries_spec(state.log, "energy", "バッテリ", height=160), use_container_width=True)
            with g2:
                st.vega_lite_chart(vega_timeseries_spec(state.log, "data_downlinked", "DL量（各ターン）", height=160), use_container_width=True)

    with t2:
        st.subheader("スコア内訳（見える化）")
        st.vega_lite_chart(vega_breakdown_bar(bd), use_container_width=True)
        st.write({k: bd[k] for k in ["距離ペナルティ", "マヌーバペナルティ", "データボーナス", "電力ボーナス", "データ損失ペナルティ", "ブラックアウトペナルティ"]})

    st.divider()
    b1, b2 = st.columns([1.2, 1.0])
    with b1:
        st.info("コツ：通信OK（緑）を維持しつつ、白い予測点が“水色の範囲内でターゲットへ寄る”ように調整。SRPは弱いので、当たりは“少しずつ”。")
    with b2:
        if st.button("🔁 もう一回（リセット）", use_container_width=True):
            reset()


if page == "Play":
    render_play()
else:
    render_result()

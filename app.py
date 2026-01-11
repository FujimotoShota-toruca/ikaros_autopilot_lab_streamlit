# IKAROS：B-plane ダーツ（適応誘導オペレーション）
# Streamlit + Vega-Lite (direct spec)
#
# v6:
# - βin/βout 平面上に「発電/電力収支」「通信可否(=ダウンリンク可否)」を可視化するマップを追加
# - β=0 のままでも勝ててしまう問題を修正：
#     * 初期B-plane誤差（投入誤差）を与える
#     * 推定(p_est)が真値(B_true)を動かしてしまう誤りを修正（推定は物理に影響しない）
#     * 物理は「真値パラメータ p_true による制御ゲインのズレ」として表現
#
# Note: 教育用の抽象モデルであり、実機の飛行力学・運用の厳密再現ではありません。
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def l2(xy: np.ndarray) -> float:
    return float(np.linalg.norm(xy))


def cosd(deg: float) -> float:
    return math.cos(math.radians(deg))


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

    S_pre  = mat(180,  40,  -20, 140)
    S_pre2 = mat(210,  60,  -40, 170)
    S_pre3 = mat(240,  70,  -60, 190)

    S_post  = mat(520, 130,  -90, 430)
    S_post2 = mat(560, 150, -110, 460)
    S_post3 = mat(600, 170, -120, 500)
    S_post4 = mat(640, 190, -140, 520)

    return [
        Section("Section 1", S_pre,    6,  6,  True,  65, 0.45, earth_angle_bias_deg=+5),
        Section("Section 2", S_pre2,   6,  6,  True,  80, 0.50, earth_angle_bias_deg=+12),
        Section("Section 3", S_pre3,   5,  5,  True,  95, 0.55, earth_angle_bias_deg=+25),
        Section("Section 4 (NO-LINK)", S_post, 0,  0,  False, 0,  0.60, earth_angle_bias_deg=+35),
        Section("Section 5", S_post2,  18, 18, True,  45, 0.70, earth_angle_bias_deg=+18),
        Section("Section 6", S_post3,  18, 18, True,  35, 0.78, earth_angle_bias_deg=+8),
        Section("Section 7", S_post4,  15, 15, True,  30, 0.85, earth_angle_bias_deg=+2),
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
    B_obs_last: np.ndarray | None

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


def execute_section(state: GameState, cfg: GameConfig, sections: List[Section]) -> None:
    rng = np.random.default_rng()
    rng.bit_generator.state = state.rng_state["bitgen"]

    section = sections[state.k]
    plan = np.array([cfg.plan_beta_in_deg, cfg.plan_beta_out_deg], dtype=float)
    cmd = np.array([state.beta_in, state.beta_out], dtype=float)
    dβ = cmd - plan

    comm_ok = comm_available(float(cmd[0]), float(cmd[1]), section, cfg, state.energy)
    if not comm_ok:
        dβ = np.array([0.0, 0.0], dtype=float)

    dβ[0] = clamp(dβ[0], -section.dbeta_in_max, section.dbeta_in_max)
    dβ[1] = clamp(dβ[1], -section.dbeta_out_max, section.dbeta_out_max)

    total_deg = abs(dβ[0]) + abs(dβ[1])
    maneuvers = section.maneuvers_per_deg * total_deg
    if maneuvers > state.maneuvers_left:
        scale = 0.0 if state.maneuvers_left <= 0 else (state.maneuvers_left / max(maneuvers, 1e-9))
        dβ *= scale
        maneuvers = section.maneuvers_per_deg * (abs(dβ[0]) + abs(dβ[1]))

    state.maneuvers_left -= maneuvers

    beta_eff_val = beta_eff(float(cmd[0]), float(cmd[1])) if comm_ok else 0.0
    gen = cfg.gen_scale * max(0.0, cosd(beta_eff_val))
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
    u_est  = np.array([dβ[0] * state.p_est[0],  dβ[1] * state.p_est[1]], dtype=float)

    rcs_bias = rng.normal(0, cfg.rcs_sigma_per_sqrt_maneuver * math.sqrt(max(maneuvers, 0.0)), size=(2,))
    state.B_true = state.B_true + section.S @ u_true + rcs_bias
    state.B_est  = state.B_est  + section.S @ u_est

    B_obs = state.B_true + rng.normal(0, cfg.meas_sigma_km, size=(2,))
    state.B_obs_last = B_obs

    od_gain_eff = section.od_gain * (0.35 if state.energy < 30.0 else 1.0)
    state.p_est, state.P_cov = od_update_gains(B_obs, state.B_est, dβ, section, state, cfg, od_gain_eff)

    sigma = np.sqrt(np.diag(state.P_cov))
    dist = l2(state.B_true - cfg.target)
    ea = predicted_earth_angle_deg(float(cmd[0]), float(cmd[1]), section, cfg)

    state.log.append(
        {
            "section": section.name,
            "comm_ok": int(comm_ok),
            "earth_angle_deg": float(ea),
            "beta_in": float(cmd[0]),
            "beta_out": float(cmd[1]),
            "beta_eff_deg": float(beta_eff_val),
            "applied_dbeta_in": float(dβ[0]),
            "applied_dbeta_out": float(dβ[1]),
            "maneuvers_used": float(maneuvers),
            "maneuvers_left": float(state.maneuvers_left),
            "energy": float(state.energy),
            "data_downlinked": float(down),
            "data_buffer": float(state.data_buffer),
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

    s = 10000.0
    s -= 0.65 * dist
    s -= 0.25 * used
    s += 55.0 * state.data_downlinked
    s += 8.0 * state.energy
    s -= 25.0 * state.data_lost
    s -= 600.0 * state.blackout_count
    s = max(0.0, s)

    return s, {
        "final_distance_km": float(dist),
        "maneuvers_used": float(used),
        "energy_left": float(state.energy),
        "science_downlinked": float(state.data_downlinked),
        "data_lost": float(state.data_lost),
        "blackouts": int(state.blackout_count),
        "score": float(s),
    }


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


def vega_bplane_spec(state: GameState, cfg: GameConfig, sections: List[Section], show_truth: bool) -> Dict:
    section = sections[min(state.k, len(sections) - 1)]
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
        {"BT": float(state.B_est[0]), "BR": float(state.B_est[1]), "kind": "推定点 E（いま）"},
    ]
    if show_truth:
        pts.append({"BT": float(state.B_true[0]), "BR": float(state.B_true[1]), "kind": "真値（いま）"})
    if state.B_obs_last is not None:
        pts.append({"BT": float(state.B_obs_last[0]), "BR": float(state.B_obs_last[1]), "kind": "観測点（前ターン）"})

    all_bt = [p["BT"] for p in pts] + [p["BT"] for p in poly_vals] + [p["BT"] for p in ring_vals]
    all_br = [p["BR"] for p in pts] + [p["BR"] for p in poly_vals] + [p["BR"] for p in ring_vals]
    span = max(12000.0, max(map(abs, all_bt + [0])), max(map(abs, all_br + [0])))
    span = float(span * 1.15)

    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "height": 420,
        "encoding": {
            "x": {"field": "BT", "type": "quantitative", "title": "BT [km]", "scale": {"domain": [-span, span]}},
            "y": {"field": "BR", "type": "quantitative", "title": "BR [km]", "scale": {"domain": [-span, span]}},
        },
        "layer": [
            {"data": {"values": ring_vals}, "mark": {"type": "line", "opacity": 0.25},
             "encoding": {"order": {"field": "i", "type": "quantitative"}}},
            {"data": {"values": poly_vals}, "mark": {"type": "line", "opacity": 0.35},
             "encoding": {"order": {"field": "idx", "type": "quantitative"}}},
            {"data": {"values": pts}, "mark": {"type": "point", "filled": True, "size": 110},
             "encoding": {
                 "shape": {"field": "kind", "type": "nominal", "legend": {"title": ""}},
                 "tooltip": [
                     {"field": "kind", "type": "nominal"},
                     {"field": "BT", "type": "quantitative", "format": ".0f"},
                     {"field": "BR", "type": "quantitative", "format": ".0f"},
                 ],
             }},
            {"data": {"values": pts}, "mark": {"type": "text", "align": "left", "dx": 8, "dy": -8},
             "encoding": {"text": {"field": "kind", "type": "nominal"}}},
        ],
        "config": {"axis": {"labelFontSize": 12, "titleFontSize": 12}, "view": {"stroke": None}},
    }


def build_beta_map_data(section: Section, cfg: GameConfig, energy: float, step: float = 2.5) -> List[Dict]:
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


def vega_beta_map_spec(vals: List[Dict], title: str, color_field: str, color_title: str, point: Tuple[float, float]) -> Dict:
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": {"text": title, "fontSize": 14},
        "width": 340,
        "height": 340,
        "layer": [
            {
                "data": {"values": vals},
                "mark": {"type": "rect"},
                "encoding": {
                    "x": {"field": "beta_in", "type": "quantitative", "title": "βin [deg]"},
                    "y": {"field": "beta_out", "type": "quantitative", "title": "βout [deg]"},
                    "color": {"field": color_field, "type": "quantitative", "title": color_title},
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
                },
            },
            {
                "data": {"values": vals},
                "transform": [{"filter": "datum.comm_ok == 0"}],
                "mark": {"type": "rect", "opacity": 0.28},
                "encoding": {
                    "x": {"field": "beta_in", "type": "quantitative"},
                    "y": {"field": "beta_out", "type": "quantitative"},
                    "color": {"value": "black"},
                },
            },
            {
                "data": {"values": [{"beta_in": float(point[0]), "beta_out": float(point[1])}]},
                "mark": {"type": "point", "filled": True, "size": 120},
                "encoding": {
                    "x": {"field": "beta_in", "type": "quantitative"},
                    "y": {"field": "beta_out", "type": "quantitative"},
                    "color": {"value": "white"},
                },
            },
        ],
        "config": {"view": {"stroke": None}},
    }


def vega_timeseries_spec(log: List[Dict], y_field: str, y_title: str, height: int = 140) -> Dict:
    vals = [{"turn": i + 1, "section": r.get("section", f"{i+1}"), "y": float(r.get(y_field, 0.0))} for i, r in enumerate(log)]
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"values": vals},
        "height": height,
        "mark": {"type": "line", "point": True},
        "encoding": {
            "x": {"field": "turn", "type": "quantitative", "title": "ターン", "tickMinStep": 1},
            "y": {"field": "y", "type": "quantitative", "title": y_title},
            "tooltip": [{"field": "section", "type": "nominal"}, {"field": "y", "type": "quantitative"}],
        },
        "config": {"view": {"stroke": None}},
    }


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="IKAROS B-plane Darts", layout="wide")
st.title("🎯 IKAROS：B-plane ダーツ（適応誘導オペレーション）")
st.caption("βin/βoutで“当てる”＋「電力・通信・データ」。β=0放置では投入誤差が消えないようにした版。")

sections = build_sections()
cfg = GameConfig()

with st.sidebar:
    st.header("設定")
    seed = st.number_input("シード（同じ問題を再現）", min_value=1, max_value=999999, value=42, step=1)
    show_truth = st.toggle("先生モード：真値を表示", value=False)
    st.divider()
    st.markdown("**学習ポイント**")
    st.markdown(
        """
- SRPは弱く、可制御性は小さい（思ったほど動かない）
- 投入誤差がある → “何もしない”では当たらない
- 通信ウィンドウと電力で「操作そのもの」が縛られる
"""
    )

seed_int = int(seed)
if "bplane_state_v6" not in st.session_state or st.session_state.get("bplane_seed_v6") != seed_int:
    st.session_state.bplane_state_v6 = init_game(cfg, sections, seed=seed_int)
    st.session_state.bplane_seed_v6 = seed_int

state: GameState = st.session_state.bplane_state_v6


def rerun():
    (st.rerun() if hasattr(st, "rerun") else st.experimental_rerun())


def reset():
    st.session_state.bplane_state_v6 = init_game(cfg, sections, seed=seed_int)
    rerun()


st.progress(min(1.0, state.k / len(sections)))
st.write(f"進捗：**{state.k}/{len(sections)}** セクション完了（全7）")

left, right = st.columns([1.65, 1.0], gap="large")
sec = sections[min(state.k, len(sections) - 1)]

with right:
    st.subheader("βin×βout マップ")
    st.caption("色=電力収支 or ダウンリンク量。黒い膜=通信NG（コマンド送れない/データ下ろせない）")
    beta_vals = build_beta_map_data(sec, cfg, energy=state.energy, step=2.5)

    m1, m2 = st.columns(2)
    with m1:
        st.vega_lite_chart(vega_beta_map_spec(beta_vals, "電力収支", "net", "電力収支", (state.beta_in, state.beta_out)),
                           use_container_width=True)
    with m2:
        st.vega_lite_chart(vega_beta_map_spec(beta_vals, "ダウンリンク量", "downlink", "DL量", (state.beta_in, state.beta_out)),
                           use_container_width=True)

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

    comm_preview = comm_available(bi, bo, sec, cfg, state.energy)
    ea_preview = predicted_earth_angle_deg(bi, bo, sec, cfg)
    be = beta_eff(bi, bo)
    gen_preview = cfg.gen_scale * max(0.0, cosd(be))
    cost_preview = cfg.base_load + (cfg.comm_cost if comm_preview else 0.0)
    net_preview = gen_preview - cost_preview

    st.subheader("運用ステータス（このβだと…）")
    c1, c2 = st.columns(2)
    c1.metric("バッテリ", f"{state.energy:.0f}/{cfg.energy_max:.0f}")
    c2.metric("βeff", f"{be:.1f}°")

    c3, c4 = st.columns(2)
    c3.metric("地球角", f"{ea_preview:.1f}°", help=f"±{cfg.comm_window_deg:.0f}°以内が通信ウィンドウ")
    c4.metric("通信", "🟢OK" if comm_preview else "🔴NG")

    c5, c6, c7 = st.columns(3)
    c5.metric("発電", f"{gen_preview:.0f}")
    c6.metric("消費", f"{cost_preview:.0f}")
    c7.metric("収支", f"{net_preview:+.0f}")

    st.subheader("テレメトリ")
    tighten = (state.k + 1) >= cfg.target_tighten_section
    sigma = np.sqrt(np.diag(state.P_cov))
    st.metric("セクション", f"{state.k + 1}/7")
    st.metric("ターゲット半径", f"{(cfg.target_radius_late_km if tighten else cfg.target_radius_early_km):.0f} km")
    st.metric("残りマヌーバ予算", f"{state.maneuvers_left:.0f}")
    st.metric("データバッファ", f"{state.data_buffer:.0f}/{cfg.data_buffer_max:.0f}")
    st.metric("推定ゲイン", f"in={state.p_est[0]:.2f}±{sigma[0]:.2f}, out={state.p_est[1]:.2f}±{sigma[1]:.2f}")

    b1, b2 = st.columns(2)
    with b1:
        if st.button("▶ このセクションを実行", use_container_width=True, disabled=(state.phase == "result")):
            execute_section(state, cfg, sections)
            rerun()
    with b2:
        if st.button("🔁 リセット", use_container_width=True):
            reset()

with left:
    st.subheader("B-plane（的当て）")
    st.caption("投入誤差があるので、β=0放置では当たりません。")
    st.vega_lite_chart(vega_bplane_spec(state, cfg, sections, show_truth), use_container_width=True)

    if comm_available(state.beta_in, state.beta_out, sec, cfg, state.energy):
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

st.subheader("ログ")
if state.log:
    st.dataframe(state.log, use_container_width=True, hide_index=True)
else:
    st.caption("まだ実行していません。")

if state.phase == "result":
    st.divider()
    st.header("📊 リザルト")
    s, breakdown = score_game(state, cfg)
    st.subheader(f"スコア：{s:.0f} 点")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最終距離", f"{breakdown['final_distance_km']:.0f} km")
    c2.metric("使用マヌーバ", f"{breakdown['maneuvers_used']:.0f}")
    c3.metric("データ下ろし", f"{breakdown['science_downlinked']:.0f}")
    c4.metric("電力残", f"{breakdown['energy_left']:.0f}")
    st.write("内訳", breakdown)

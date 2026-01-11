# IKAROS：B-plane ダーツ（適応誘導オペレーション）
# Streamlit + Vega-Lite (direct spec)  ✅ "全部Vega"
#
# v4 fixes:
# - Avoid Altair SchemaValidationError by using st.vega_lite_chart with explicit Vega-Lite specs.
# - Keep Python 3.13 dataclass compatibility (default_factory for np.ndarray).
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st


# -----------------------------
# Helpers
# -----------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def l2(xy: np.ndarray) -> float:
    return float(np.linalg.norm(xy))


# -----------------------------
# Model
# -----------------------------
@dataclass
class Section:
    name: str
    S: np.ndarray           # km/deg
    H: np.ndarray           # km per fractional error
    dbeta_in_max: float
    dbeta_out_max: float
    comm: bool
    maneuvers_per_deg: float
    od_gain: float


def build_sections() -> List[Section]:
    def mat(a, b, c, d):
        return np.array([[a, b], [c, d]], dtype=float)

    # Control authority grows after NO-LINK (abstracted)
    S_pre  = mat(180,  40,  -20, 140)
    S_pre2 = mat(210,  60,  -40, 170)
    S_pre3 = mat(240,  70,  -60, 190)

    S_post  = mat(520, 130,  -90, 430)
    S_post2 = mat(560, 150, -110, 460)
    S_post3 = mat(600, 170, -120, 500)
    S_post4 = mat(640, 190, -140, 520)

    # Parameter sensitivity (uncertainty)
    H_pre  = mat( 6000,  2500, 12000,  7000)
    H_post = mat( 4500,  2000,  9000,  5200)

    return [
        Section("Section 1", S_pre,  H_pre,   6,  6,  True,  65, 0.45),
        Section("Section 2", S_pre2, H_pre,   6,  6,  True,  80, 0.50),
        Section("Section 3", S_pre3, H_pre,   5,  5,  True,  95, 0.55),
        Section("Section 4 (NO-LINK)", S_post, H_post, 0, 0, False, 0, 0.60),
        Section("Section 5", S_post2, H_post, 18, 18, True, 45, 0.70),
        Section("Section 6", S_post3, H_post, 18, 18, True, 35, 0.78),
        Section("Section 7", S_post4, H_post, 15, 15, True, 30, 0.85),
    ]


@dataclass
class GameConfig:
    # Python 3.13 dataclasses require default_factory for mutable defaults
    target: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0], dtype=float))  # BT, BR [km]

    target_radius_early_km: float = 9000.0
    target_radius_late_km: float = 2000.0
    target_tighten_section: int = 5  # 1-indexed

    sigma_area0: float = 0.10
    sigma_spec0: float = 0.08
    meas_sigma_km: float = 500.0

    rcs_sigma_per_sqrt_maneuver: float = 30.0
    maneuver_budget: float = 6000.0

    plan_beta_in_deg: float = 0.0
    plan_beta_out_deg: float = 0.0


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


def init_game(cfg: GameConfig, sections: List[Section], seed: int) -> GameState:
    rng = np.random.default_rng(seed)

    p_true = np.array(
        [1.0 + rng.normal(0, cfg.sigma_area0), 1.0 + rng.normal(0, cfg.sigma_spec0)],
        dtype=float,
    )
    p_est = np.array([1.0, 1.0], dtype=float)
    P_cov = np.diag([cfg.sigma_area0**2, cfg.sigma_spec0**2])

    B_est = cfg.target.copy()
    B_true = B_est + sections[0].H @ (p_true - p_est)

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
    )


def od_update(state: GameState, section: Section, cfg: GameConfig, rng: np.random.Generator):
    H = section.H
    y = state.B_true + rng.normal(0, cfg.meas_sigma_km, size=(2,))
    r = y - state.B_est

    R = np.eye(2) * (cfg.meas_sigma_km**2)
    P = state.P_cov
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    K_eff = section.od_gain * K

    dp = K_eff @ r
    p_est_new = state.p_est + dp

    I = np.eye(2)
    P_new = (I - K_eff @ H) @ P @ (I - K_eff @ H).T + K_eff @ R @ K_eff.T
    return y, p_est_new, P_new


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


def execute_section(state: GameState, cfg: GameConfig, sections: List[Section]) -> None:
    rng = np.random.default_rng()
    rng.bit_generator.state = state.rng_state["bitgen"]

    section = sections[state.k]
    plan = np.array([cfg.plan_beta_in_deg, cfg.plan_beta_out_deg], dtype=float)
    cmd = np.array([state.beta_in, state.beta_out], dtype=float)
    dβ = cmd - plan

    # NO-LINK: command locked
    if not section.comm:
        dβ = np.array([0.0, 0.0], dtype=float)

    # corridor clamp
    dβ[0] = clamp(dβ[0], -section.dbeta_in_max, section.dbeta_in_max)
    dβ[1] = clamp(dβ[1], -section.dbeta_out_max, section.dbeta_out_max)

    # RCS maneuvers / budget
    total_deg = abs(dβ[0]) + abs(dβ[1])
    maneuvers = section.maneuvers_per_deg * total_deg
    if maneuvers > state.maneuvers_left:
        scale = 0.0 if state.maneuvers_left <= 0 else (state.maneuvers_left / max(maneuvers, 1e-9))
        dβ *= scale
        total_deg = abs(dβ[0]) + abs(dβ[1])
        maneuvers = section.maneuvers_per_deg * total_deg

    state.maneuvers_left -= maneuvers

    # RCS bias grows with maneuvers (abstract)
    rcs_bias = rng.normal(0, cfg.rcs_sigma_per_sqrt_maneuver * math.sqrt(max(maneuvers, 0.0)), size=(2,))

    # Dynamics update (abstracted)
    state.B_est = state.B_est + section.S @ dβ
    state.B_true = state.B_est + section.H @ (state.p_true - state.p_est) + rcs_bias

    # OD update
    y, p_est_new, P_new = od_update(state, section, cfg, rng)
    state.B_obs_last = y
    state.p_est = p_est_new
    state.P_cov = P_new

    sigma = np.sqrt(np.diag(state.P_cov))
    dist = l2(state.B_true - cfg.target)

    state.log.append(
        {
            "section": section.name,
            "comm": section.comm,
            "cmd_beta_in": float(cmd[0]),
            "cmd_beta_out": float(cmd[1]),
            "applied_dbeta_in": float(dβ[0]),
            "applied_dbeta_out": float(dβ[1]),
            "maneuvers_used": float(maneuvers),
            "maneuvers_left": float(state.maneuvers_left),
            "BT_true_km": float(state.B_true[0]),
            "BR_true_km": float(state.B_true[1]),
            "dist_to_target_km": float(dist),
            "sigma_area": float(sigma[0]),
            "sigma_spec": float(sigma[1]),
        }
    )

    state.k += 1
    if state.k >= len(sections):
        state.phase = "result"

    state.rng_state["bitgen"] = rng.bit_generator.state


def score_game(state: GameState, cfg: GameConfig):
    dist = l2(state.B_true - cfg.target)
    used = cfg.maneuver_budget - state.maneuvers_left
    sigma = np.sqrt(np.diag(state.P_cov))
    est_bonus = 2000.0 / max(200.0, (sigma[0] * 10000 + sigma[1] * 10000))
    s = 10000.0 - 0.65 * dist - 0.25 * used + 1200.0 * est_bonus
    s = max(0.0, s)
    return s, {
        "final_distance_km": dist,
        "maneuvers_used": used,
        "maneuvers_left": state.maneuvers_left,
        "sigma_area": float(sigma[0]),
        "sigma_spec": float(sigma[1]),
        "est_bonus": float(est_bonus),
        "score": float(s),
    }


# -----------------------------
# Vega-Lite chart builders (dict specs)
# -----------------------------
def vega_timeline_spec(state: GameState, sections: List[Section]) -> Dict:
    rows = []
    for i, s in enumerate(sections):
        rows.append({
            "sec": i + 1,
            "label": s.name,
            "status": "現在" if i == state.k else ("完了" if i < state.k else "未"),
            "row": "timeline"
        })

    return {
        "data": {"values": rows},
        "height": 40,
        "mark": {"type": "rect"},
        "encoding": {
            "x": {"field": "sec", "type": "ordinal", "title": "セクション（2週間単位）"},
            "y": {"field": "row", "type": "nominal", "axis": None, "title": None},
            "color": {"field": "status", "type": "nominal", "legend": None},
            "tooltip": [
                {"field": "label", "type": "nominal"},
                {"field": "status", "type": "nominal"},
            ],
        },
        "config": {"view": {"stroke": None}},
    }


def vega_bplane_spec(state: GameState, cfg: GameConfig, sections: List[Section], preview_beta: Tuple[float, float], show_truth: bool) -> Dict:
    section = sections[min(state.k, len(sections) - 1)]
    plan = np.array([cfg.plan_beta_in_deg, cfg.plan_beta_out_deg], dtype=float)
    preview = np.array(preview_beta, dtype=float)

    dβ = preview - plan
    if not section.comm:
        dβ = np.array([0.0, 0.0], dtype=float)

    dβ[0] = clamp(dβ[0], -section.dbeta_in_max, section.dbeta_in_max)
    dβ[1] = clamp(dβ[1], -section.dbeta_out_max, section.dbeta_out_max)

    B_preview = state.B_est + section.S @ dβ

    # Uncertainty ellipse (axis-aligned) from mapped covariance
    P = state.P_cov
    H = section.H
    CovB = H @ P @ H.T + np.eye(2) * (cfg.meas_sigma_km**2) * 0.25
    rad_BT = float(max(300.0, math.sqrt(max(CovB[0, 0], 1.0))))
    rad_BR = float(max(300.0, math.sqrt(max(CovB[1, 1], 1.0))))

    # Controllability polygon centered at estimate
    poly = compute_controllability_polygon(section) + state.B_est.reshape(1, 2)
    poly_vals = [{"BT": float(p[0]), "BR": float(p[1]), "idx": i} for i, p in enumerate(poly)]

    pts = [
        {"BT": float(cfg.target[0]), "BR": float(cfg.target[1]), "kind": "ターゲット中心"},
        {"BT": float(state.B_est[0]), "BR": float(state.B_est[1]), "kind": "推定点 E（いま）"},
        {"BT": float(B_preview[0]), "BR": float(B_preview[1]), "kind": "プレビュー（このコマンド）"},
    ]
    if state.B_obs_last is not None:
        pts.append({"BT": float(state.B_obs_last[0]), "BR": float(state.B_obs_last[1]), "kind": "OD点（前ターン）"})
    if show_truth:
        pts.append({"BT": float(state.B_true[0]), "BR": float(state.B_true[1]), "kind": "真値（先生）"})

    tighten = (state.k + 1) >= cfg.target_tighten_section
    target_r = cfg.target_radius_late_km if tighten else cfg.target_radius_early_km

    ring_vals = []
    for i in range(65):
        th = 2 * math.pi * i / 64
        ring_vals.append({"BT": float(cfg.target[0] + target_r * math.cos(th)),
                          "BR": float(cfg.target[1] + target_r * math.sin(th)),
                          "i": i})

    ell_vals = []
    for i in range(65):
        th = 2 * math.pi * i / 64
        ell_vals.append({"BT": float(state.B_est[0] + rad_BT * math.cos(th)),
                         "BR": float(state.B_est[1] + rad_BR * math.sin(th)),
                         "i": i})

    # View span
    all_bt = [p["BT"] for p in pts] + [p["BT"] for p in poly_vals]
    all_br = [p["BR"] for p in pts] + [p["BR"] for p in poly_vals]
    span = max(12000.0, max(map(abs, all_bt + [0])), max(map(abs, all_br + [0])))
    span = float(span * 1.15)

    return {
        "height": 420,
        "encoding": {
            "x": {"field": "BT", "type": "quantitative", "title": "BT [km]", "scale": {"domain": [-span, span]}},
            "y": {"field": "BR", "type": "quantitative", "title": "BR [km]", "scale": {"domain": [-span, span]}},
        },
        "layer": [
            {
                "data": {"values": ring_vals},
                "mark": {"type": "line", "opacity": 0.35},
                "encoding": {"order": {"field": "i", "type": "quantitative"}},
            },
            {
                "data": {"values": ell_vals},
                "mark": {"type": "area", "opacity": 0.10},
                "encoding": {"order": {"field": "i", "type": "quantitative"}},
            },
            {
                "data": {"values": poly_vals},
                "mark": {"type": "line", "opacity": 0.35},
                "encoding": {"order": {"field": "idx", "type": "quantitative"}},
            },
            {
                "data": {"values": pts},
                "mark": {"type": "point", "filled": True, "size": 120},
                "encoding": {
                    "shape": {"field": "kind", "type": "nominal", "legend": {"title": ""}},
                    "tooltip": [
                        {"field": "kind", "type": "nominal"},
                        {"field": "BT", "type": "quantitative", "format": ".0f"},
                        {"field": "BR", "type": "quantitative", "format": ".0f"},
                    ],
                },
            },
            {
                "data": {"values": pts},
                "mark": {"type": "text", "align": "left", "dx": 8, "dy": -8},
                "encoding": {"text": {"field": "kind", "type": "nominal"}},
            },
        ],
        "config": {"axis": {"labelFontSize": 12, "titleFontSize": 12}},
    }


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="IKAROS B-plane Darts", layout="wide")
st.title("🎯 IKAROS：B-plane ダーツ（適応誘導オペレーション）")
st.caption("2週間=1ターン。βin/βout → ODで推定更新 → 後半勝負、を体感する“運用ゲーム”。")

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
- SRPの可制御性は小さい
- パラメータ不確かさが大きい
- no-linkで操作不能区間がある
- ODで推定が改善 → 後半が勝負
- RCSの副作用（マヌーバ）
"""
    )

seed_int = int(seed)
if "bplane_state_v4" not in st.session_state or st.session_state.get("bplane_seed_v4") != seed_int:
    st.session_state.bplane_state_v4 = init_game(cfg, sections, seed=seed_int)
    st.session_state.bplane_seed_v4 = seed_int

state: GameState = st.session_state.bplane_state_v4


def rerun():
    (st.rerun() if hasattr(st, "rerun") else st.experimental_rerun())


def reset():
    st.session_state.bplane_state_v4 = init_game(cfg, sections, seed=seed_int)
    rerun()


# Timeline
st.vega_lite_chart(vega_timeline_spec(state, sections), use_container_width=True)

with st.expander("ノミナル（計画）とは？", expanded=False):
    st.markdown(
        """
- **ノミナル（計画）**：事前に作る参照（目標）状態。ここでは **計画β=0°** を基準にします。
- **操作**：βin/βout を計画からずらす（Δβ）
- **OD**：観測で“実際の当たり”がわかり、帆パラメータ推定が更新される
"""
    )

left, right = st.columns([1.6, 1.0], gap="large")
sec = sections[min(state.k, len(sections) - 1)]

with right:
    st.subheader("コマンド（βin / βout）")
    if state.phase != "result" and not sec.comm:
        st.warning("このセクションは NO-LINK：コマンド送信不可（Δβ=0固定）。")

    st.caption("このセクションの制約（簡略）")
    st.write(
        {
            "Δβin 最大": f"±{sec.dbeta_in_max:.0f}°",
            "Δβout 最大": f"±{sec.dbeta_out_max:.0f}°",
            "maneuvers/deg": f"{sec.maneuvers_per_deg:.0f}",
            "OD品質": f"{sec.od_gain:.2f}",
        }
    )

    cA, cB = st.columns(2)
    with cA:
        bi = st.slider("βin [deg]", -35.0, 35.0, float(state.beta_in), 1.0, disabled=(state.phase == "result"))
        bi = float(st.number_input("βin 直打ち", -90.0, 90.0, bi, 1.0, disabled=(state.phase == "result")))
    with cB:
        bo = st.slider("βout [deg]", -35.0, 35.0, float(state.beta_out), 1.0, disabled=(state.phase == "result"))
        bo = float(st.number_input("βout 直打ち", -90.0, 90.0, bo, 1.0, disabled=(state.phase == "result")))

    state.beta_in = bi
    state.beta_out = bo

    st.subheader("テレメトリ")
    sigma = np.sqrt(np.diag(state.P_cov))
    tighten = (state.k + 1) >= cfg.target_tighten_section
    st.metric("セクション", f"{state.k + 1}/7")
    st.metric("ターゲット半径", f"{(cfg.target_radius_late_km if tighten else cfg.target_radius_early_km):.0f} km")
    st.metric("推定誤差(面積)", f"±{sigma[0] * 100:.1f}%")
    st.metric("推定誤差(反射率)", f"±{sigma[1] * 100:.1f}%")
    st.metric("残りマヌーバ予算", f"{state.maneuvers_left:.0f}")

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
    st.caption("雲=不確かさ / 四角=このセクションで動かせる範囲 / プレビュー=今の入力βを実行した場合の予測")
    st.vega_lite_chart(vega_bplane_spec(state, cfg, sections, (state.beta_in, state.beta_out), show_truth), use_container_width=True)
    st.info("コツ：序盤は雲（不確かさ）が大きく、当てに行っても外れがち。ODで雲が小さくなってから後半で勝負。")

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
    c1, c2, c3 = st.columns(3)
    c1.metric("最終距離", f"{breakdown['final_distance_km']:.0f} km")
    c2.metric("使用マヌーバ", f"{breakdown['maneuvers_used']:.0f}")
    c3.metric("推定誤差(面積/反射率)", f"±{breakdown['sigma_area'] * 100:.1f}% / ±{breakdown['sigma_spec'] * 100:.1f}%")
    st.write("内訳", breakdown)
    st.success("SRPは小さいので、可制御性より推定誤差や運用制約が効いてくる。だから適応誘導（推定→更新→後半勝負）が大事。")

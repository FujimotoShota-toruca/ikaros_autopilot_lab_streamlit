"""
ゲームの状態遷移（運用・OD・リソース管理）

v-angle 版のポイント：
- 発電/通信を “角度モデル” に統一（core/attitude.py）
  - α = angle(n, s) : 発電
  - γ = angle(n, e) : 通信
- 通信は「コマンド可否」ではなく「データ下ろし可否」として扱う
  （“指向を合わせれば通信できる” をゲームとして素直にする）
- NO-LINK セクションだけは「操作できない」演出として Δβ=0 に固定する
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import GameConfig, EphemConfig
from .attitude import (
    angle_deg,
    comm_ok_from_gamma,
    compute_dirs,
    downlink_rate_from_gamma,
    power_from_alpha,
    sail_normal_from_beta,
)


# -----------------------------
# 共通ユーティリティ
# -----------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def l2(xy: np.ndarray) -> float:
    return float(np.linalg.norm(xy))


# -----------------------------
# エフェメリス（超簡易）
# -----------------------------
def pos_au(r: float, theta_deg: float) -> np.ndarray:
    th = math.radians(theta_deg)
    return np.array([r * math.cos(th), r * math.sin(th)], dtype=float)


def ephem_at_day(t_day: float, eph: EphemConfig) -> Dict[str, np.ndarray]:
    th_e = eph.theta_e0 + eph.omega_earth * t_day
    th_v = eph.theta_v0 + eph.omega_venus * t_day
    return {
        "earth": pos_au(eph.r_earth, th_e),
        "venus": pos_au(eph.r_venus, th_v),
        "th_e": th_e,
        "th_v": th_v,
    }


def sc_nominal_at_fraction(u: float, eph: EphemConfig) -> Dict[str, np.ndarray]:
    """
    ノミナル（計画）の探査機位置（概念）を、u∈[0,1] で連続に定義。

    - 角度：Earth(t=0) → Venus(t=t_end) を線形補間
    - 半径：1AU → 0.723AU を線形補間

    最終u=1で「金星位置（t_end）」に一致するように作っている。
    """
    u = clamp(float(u), 0.0, 1.0)

    e0 = ephem_at_day(0.0, eph)
    vend = ephem_at_day(eph.t_end_day, eph)

    th_start = float(e0["th_e"])
    th_target = float(vend["th_v"])

    th_sc = (1.0 - u) * th_start + u * th_target
    r_sc = eph.r_earth + (eph.r_venus - eph.r_earth) * u
    return {"sc": pos_au(r_sc, th_sc), "th_sc": th_sc, "r_sc": r_sc}


def sc_nominal_at_index(k: int, n: int, eph: EphemConfig) -> Dict[str, np.ndarray]:
    u = 0.0 if n <= 1 else float(k) / float(n - 1)
    return sc_nominal_at_fraction(u, eph)


# -----------------------------
# セクション定義（ゲームのステージ）
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

    # ここから感度が一気に変わる＝フライバイ後っぽい演出
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


# -----------------------------
# ゲーム状態
# -----------------------------
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
    rng = np.random.default_rng(int(seed))

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
        beta_in=float(cfg.plan_beta_in_deg),
        beta_out=float(cfg.plan_beta_out_deg),
        maneuvers_left=float(cfg.maneuver_budget),
        log=[],
        phase="play",
        rng_state={"seed": int(seed), "bitgen": rng.bit_generator.state},
        energy=float(cfg.energy_init),
        data_buffer=0.0,
        data_downlinked=0.0,
        data_lost=0.0,
        blackout_count=0,
    )


# -----------------------------
# 角度の計算（このゲームの“本質”）
# -----------------------------
def current_positions(state: GameState, cfg: GameConfig, sections: List[Section]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """現在セクションの SC/地球/金星 位置（2D, AU）"""
    n = len(sections)
    sec = sections[min(state.k, n - 1)]
    e = ephem_at_day(sec.t_day, cfg.eph)
    r_sc = sc_nominal_at_index(min(state.k, n - 1), n, cfg.eph)["sc"]
    return r_sc, e["earth"], e["venus"]


def alpha_gamma_deg(beta_in: float, beta_out: float, state: GameState, cfg: GameConfig, sections: List[Section]) -> Tuple[float, float]:
    """帆法線 n を作って、α=angle(n,s), γ=angle(n,e) を返す"""
    r_sc, r_e, _ = current_positions(state, cfg, sections)
    s_hat, e_hat = compute_dirs(r_sc, r_e)
    n_hat = sail_normal_from_beta(beta_in, beta_out, s_hat)
    alpha = angle_deg(n_hat, s_hat)
    gamma = angle_deg(n_hat, e_hat)
    return alpha, gamma


def comm_ok(beta_in: float, beta_out: float, state: GameState, cfg: GameConfig, sections: List[Section]) -> bool:
    """通信OK（角度モデル）。NO-LINKは常にNG（演出）"""
    sec = sections[min(state.k, len(sections) - 1)]
    if not sec.uplink_possible:
        return False
    alpha, gamma = alpha_gamma_deg(beta_in, beta_out, state, cfg, sections)
    return comm_ok_from_gamma(gamma, cfg.comm_cone_deg, state.energy, cfg.energy_min_for_comm)


def power_gen(beta_in: float, beta_out: float, state: GameState, cfg: GameConfig, sections: List[Section]) -> Tuple[float, float, float]:
    """発電量（Pgen）と、参考としてα,γも返す"""
    alpha, gamma = alpha_gamma_deg(beta_in, beta_out, state, cfg, sections)
    Pgen = power_from_alpha(alpha, cfg.gen_scale, cfg.gen_cos_k)
    return Pgen, alpha, gamma


# -----------------------------
# OD更新（概念）
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
# コマンド → Δβ（NO-LINKなら固定）
# -----------------------------
def clamp_dbeta(dβ: np.ndarray, section: Section) -> np.ndarray:
    d = dβ.copy()
    d[0] = clamp(d[0], -section.dbeta_in_max, section.dbeta_in_max)
    d[1] = clamp(d[1], -section.dbeta_out_max, section.dbeta_out_max)
    return d


def commanded_dbeta(state: GameState, cfg: GameConfig, section: Section) -> np.ndarray:
    """Δβ：通信では縛らない。NO-LINKだけはΔβ=0"""
    if not section.uplink_possible:
        return np.array([0.0, 0.0], dtype=float)

    plan = np.array([cfg.plan_beta_in_deg, cfg.plan_beta_out_deg], dtype=float)
    cmd = np.array([state.beta_in, state.beta_out], dtype=float)
    dβ = cmd - plan
    return clamp_dbeta(dβ, section)


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
    dβ0 = commanded_dbeta(state, cfg, section)
    dβ, maneuvers, limited = scale_by_budget(dβ0, section, state.maneuvers_left)

    u_est = np.array([dβ[0] * state.p_est[0], dβ[1] * state.p_est[1]], dtype=float)
    B_pred = state.B_est + section.S @ u_est

    G = section.S @ np.diag([float(dβ[0]), float(dβ[1])])
    cov_gain = G @ state.P_cov @ G.T

    sig_rcs = cfg.rcs_sigma_per_sqrt_maneuver * math.sqrt(max(maneuvers, 0.0))
    cov_rcs = np.eye(2) * (sig_rcs**2)
    cov_pred = cov_gain + cov_rcs

    if not np.isfinite(cov_pred).all():
        cov_pred = np.eye(2) * (cfg.meas_sigma_km**2)

    return {"dβ": dβ, "maneuvers": maneuvers, "budget_limited": bool(limited), "B_pred": B_pred, "cov_pred": cov_pred}


def execute_section(state: GameState, cfg: GameConfig, sections: List[Section]) -> None:
    rng = np.random.default_rng()
    rng.bit_generator.state = state.rng_state["bitgen"]

    section = sections[state.k]
    pv = preview_next(state, cfg, sections, section)
    dβ = pv["dβ"]
    maneuvers = pv["maneuvers"]

    state.maneuvers_left -= maneuvers

    # 発電・通信（角度モデル）
    Pgen, alpha, gamma = power_gen(state.beta_in, state.beta_out, state, cfg, sections)
    ok_comm = comm_ok(state.beta_in, state.beta_out, state, cfg, sections)

    # 電力：発電 - 消費
    cost = cfg.base_load + cfg.maneuver_energy_scale * maneuvers + (cfg.comm_cost if ok_comm else 0.0)
    state.energy = clamp(state.energy + Pgen - cost, 0.0, cfg.energy_max)
    if state.energy <= 1e-6:
        state.blackout_count += 1

    # 科学データ：溜める＆下ろす
    collected = cfg.data_collect_hi if state.energy >= 40.0 else cfg.data_collect_lo
    state.data_buffer += collected
    overflow = max(0.0, state.data_buffer - cfg.data_buffer_max)
    if overflow > 0:
        state.data_lost += overflow
        state.data_buffer = cfg.data_buffer_max

    down = 0.0
    if ok_comm:
        down = min(state.data_buffer, downlink_rate_from_gamma(gamma, cfg.comm_cone_deg, cfg.data_downlink_cap))
        state.data_buffer -= down
        state.data_downlinked += down

    # B-plane：真値を進める
    u_true = np.array([dβ[0] * state.p_true[0], dβ[1] * state.p_true[1]], dtype=float)
    u_est = np.array([dβ[0] * state.p_est[0], dβ[1] * state.p_est[1]], dtype=float)

    rcs_bias = rng.normal(0, cfg.rcs_sigma_per_sqrt_maneuver * math.sqrt(max(maneuvers, 0.0)), size=(2,))
    state.B_true = state.B_true + section.S @ u_true + rcs_bias
    state.B_est = state.B_est + section.S @ u_est

    # 観測（OD用）
    B_obs = state.B_true + rng.normal(0, cfg.meas_sigma_km, size=(2,))
    state.B_obs_last = B_obs

    # OD更新
    od_gain_eff = section.od_gain * (0.35 if state.energy < 30.0 else 1.0)
    state.p_est, state.P_cov = od_update_gains(B_obs, state.B_est, dβ, section, state, cfg, od_gain_eff)

    # ログ
    state.log.append(
        {
            "turn": int(state.k + 1),
            "section": section.name,
            "t_day": float(section.t_day),
            "alpha_sun_deg": float(alpha),
            "gamma_earth_deg": float(gamma),
            "comm_ok": int(ok_comm),
            "beta_in": float(state.beta_in),
            "beta_out": float(state.beta_out),
            "maneuvers_used": float(maneuvers),
            "maneuvers_left": float(state.maneuvers_left),
            "Pgen": float(Pgen),
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
# スコア計算（教材用：わかりやすい線形）
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

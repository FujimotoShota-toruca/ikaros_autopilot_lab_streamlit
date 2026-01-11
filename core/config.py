"""
設定まわり（ゲームパラメータ）

この教材は「実フライトの正確な数値」よりも、
  - SRP（太陽光圧）の可制御性が小さい
  - 通信ウィンドウや発電とのトレードオフがある
  - OD（軌道決定）の推定誤差が運用に効く
という“本質”を、ゲームとして体験してもらうことを優先しています。

そのため、モデルは意図的に単純化されています。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


# -----------------------------
# エフェメリス（超簡易）
# -----------------------------
@dataclass
class EphemConfig:
    # 公転角速度（deg/day）：円軌道の近似
    omega_earth: float = 360.0 / 365.25
    omega_venus: float = 360.0 / 224.7

    # 半径（AU）：円軌道の近似
    r_earth: float = 1.0
    r_venus: float = 0.723

    # 初期位相（deg）
    theta_e0: float = 0.0
    theta_v0: float = 35.0

    # “到着時刻”（最終セクションの時刻と一致させる）
    t_end_day: float = 170.0


# -----------------------------
# ゲーム設定（いじるならここ）
# -----------------------------
@dataclass
class GameConfig:
    # B-plane上のターゲット中心（km）
    target: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0], dtype=float))

    # 序盤は大きめ、終盤は小さめの“的”にする（難易度を上げる）
    target_radius_early_km: float = 9000.0
    target_radius_late_km: float = 2000.0
    target_tighten_section: int = 5

    # 初期投入誤差（真値）と、推定値の初期誤差
    init_error_sigma_km: float = 6500.0
    init_est_sigma_km: float = 1200.0

    # SRP効果の“モデル化誤差”をパラメータ化（p_true と p_est のズレ）
    sigma_gain_in0: float = 0.10
    sigma_gain_out0: float = 0.08

    # 観測（OD）の雑音
    meas_sigma_km: float = 500.0

    # RCS（姿勢制御・マヌーバ）副作用の雑音
    rcs_sigma_per_sqrt_maneuver: float = 30.0

    # マヌーバ（燃料/反作用ホイール等）の総量予算
    maneuver_budget: float = 6000.0

    # 計画（ノミナル）のβ（ゲーム中はこれを基準にΔβを作る）
    plan_beta_in_deg: float = 0.0
    plan_beta_out_deg: float = 0.0

    # -------------------------
    # 通信・指向モデル（簡易）
    # -------------------------
    # 地球角がこの範囲内なら通信OK（deg）
    comm_window_deg: float = 20.0

    # βの指向（βpoint）で地球角がどれだけ動くか（概念係数）
    beta_point_coupling: float = 0.70

    # -------------------------
    # 電力モデル（簡易）
    # -------------------------
    energy_max: float = 200.0
    energy_init: float = 140.0
    energy_min_for_comm: float = 30.0

    # 常時消費（機器）
    base_load: float = 70.0

    # 発電スケール（βeffが大きいほど cos で落ちる）
    gen_scale: float = 90.0

    # 通信時の追加消費
    comm_cost: float = 10.0

    # マヌーバによる追加消費
    maneuver_energy_scale: float = 0.02

    # -------------------------
    # 科学データ（簡易）
    # -------------------------
    data_buffer_max: float = 60.0
    data_collect_hi: float = 12.0
    data_collect_lo: float = 4.0
    data_downlink_cap: float = 18.0

    # 予測楕円のσ（1σ / 2σ の切替などに使える）
    pred_ellipse_sigma: float = 1.0

    # エフェメリス設定
    eph: EphemConfig = field(default_factory=EphemConfig)

"""
設定まわり（ゲームパラメータ）

この教材は「実フライトの正確な再現」ではなく、
  - SRP（太陽光圧）の可制御性が小さい
  - 発電（太陽指向）と通信（地球指向）のトレードオフ
  - OD（軌道決定）で “効き” を学習する
をゲームとして体験することを目的にしています。

v-angle 版では、通信/発電を
  「帆面法線ベクトル n と、太陽方向 s・地球方向 e のなす角」
だけで決める、シンプルな定義に統一しています。
"""

from __future__ import annotations

from dataclasses import dataclass, field

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

    # 序盤/終盤で的を小さくする（難易度を上げる）
    target_radius_early_km: float = 9000.0
    target_radius_late_km: float = 2000.0
    target_tighten_section: int = 5

    # 初期投入誤差（真値）と、推定値の初期誤差
    init_error_sigma_km: float = 6500.0
    init_est_sigma_km: float = 1200.0

    # SRP効きの“モデル化誤差”をパラメータ化（p_true と p_est のズレ）
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
    # 通信（角度モデル）
    # -------------------------
    # 地球方向 e と帆法線 n のなす角 γ がこの範囲内なら通信OK（deg）
    comm_cone_deg: float = 20.0

    # -------------------------
    # 電力（角度モデル）
    # -------------------------
    # 太陽方向 s と帆法線 n のなす角 α で発電が落ちる：Pgen = P0 * max(0,cosα)^k
    gen_scale: float = 90.0
    gen_cos_k: float = 1.6  # 1.0だと緩い / 2.0だとシビア

    energy_max: float = 200.0
    energy_init: float = 140.0
    energy_min_for_comm: float = 30.0

    # 常時消費（機器）
    base_load: float = 70.0

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

    # 最大DL（通信中心で最大、端に寄ると減る：後述のモデルで滑らかにする）
    data_downlink_cap: float = 18.0

    # 予測楕円のσ（1σ / 2σ）
    pred_ellipse_sigma: float = 1.0

    # エフェメリス設定
    eph: EphemConfig = field(default_factory=EphemConfig)

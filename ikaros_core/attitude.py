"""
姿勢・幾何の共通関数（角度モデル）

狙い：
- “通信できるか？” “発電できるか？” を
  帆法線 n と、地球方向 e・太陽方向 s の “なす角” だけで決める。
- βin/βout は、n を作るための“つまみ”。
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def unit(v: np.ndarray) -> np.ndarray:
    """ゼロ割を避けつつ単位化"""
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return v / n


def angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    """2ベクトルのなす角（deg）。数値誤差でNaNにならないようclipする"""
    c = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))




def angle_bisided_deg(u: np.ndarray, v: np.ndarray) -> float:
    """2ベクトルの“両面”なす角（deg）。

    アンテナが表裏どちらにもある、という前提で
      gamma = min(angle(u,v), angle(-u,v))
    を使いたいときに用います。

    数式的には arccos(|u·v|) と等価です。
    """
    c = float(np.clip(abs(np.dot(u, v)), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))

def rot_y(deg: float) -> np.ndarray:
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], dtype=float)


def rot_z(deg: float) -> np.ndarray:
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=float)


def make_frame_from_sun(s_hat_world: np.ndarray) -> np.ndarray:
    """
    太陽方向 s_hat_world をローカルx軸にした正規直交基底を作る。
    world_vec = R @ local_vec を満たす（列ベクトルが基底）。
    """
    x = unit(s_hat_world)
    z0 = np.array([0.0, 0.0, 1.0], dtype=float)
    y = np.cross(z0, x)
    if np.linalg.norm(y) < 1e-6:
        z0 = np.array([0.0, 1.0, 0.0], dtype=float)
        y = np.cross(z0, x)
    y = unit(y)
    z = unit(np.cross(x, y))
    return np.stack([x, y, z], axis=1)


def sail_normal_from_beta(beta_in: float, beta_out: float, s_hat_world: np.ndarray) -> np.ndarray:
    """
    βin/βout から帆法線 n（世界座標）を作る。

    βpoint = (βin + βout)/2    : 符号つき（地球指向に効きやすい）
    βeff   = (|βin|+|βout|)/2  : 絶対値（太陽指向＝発電に効きやすい）
    """
    beta_point = 0.5 * (beta_in + beta_out)
    beta_eff = 0.5 * (abs(beta_in) + abs(beta_out))

    Rw = make_frame_from_sun(s_hat_world)
    s_local = np.array([1.0, 0.0, 0.0], dtype=float)

    n_local = rot_z(beta_point) @ rot_y(beta_eff) @ s_local
    return unit(Rw @ n_local)


def power_from_alpha(alpha_deg: float, P0: float, k: float) -> float:
    """発電モデル：Pgen = P0 * max(0,cosα)^k"""
    c = max(0.0, math.cos(math.radians(alpha_deg)))
    return float(P0 * (c ** float(k)))


def comm_ok_from_gamma(gamma_deg: float, gamma_max_deg: float, energy: float, Emin: float) -> bool:
    """通信OK：指向誤差（コーン）＋最低電力"""
    return (gamma_deg <= gamma_max_deg) and (energy >= Emin)


def downlink_rate_from_gamma(gamma_deg: float, gamma_max_deg: float, cap: float) -> float:
    """DL量（滑らか版）：中心ほど↑、端ほど↓"""
    if gamma_deg >= gamma_max_deg:
        return 0.0
    x = gamma_deg / max(gamma_max_deg, 1e-9)
    f = max(0.0, 1.0 - x * x)
    return float(cap * f)


def to3(v2: np.ndarray) -> np.ndarray:
    return np.array([float(v2[0]), float(v2[1]), 0.0], dtype=float)


def compute_dirs(r_sc_au_xy: np.ndarray, r_earth_au_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """2D位置（AU）→ 3D方向ベクトル（単位）"""
    r_sc = to3(r_sc_au_xy)
    r_e = to3(r_earth_au_xy)
    s_hat = unit(-r_sc)       # SC → Sun（原点）
    e_hat = unit(r_e - r_sc)  # SC → Earth
    return s_hat, e_hat

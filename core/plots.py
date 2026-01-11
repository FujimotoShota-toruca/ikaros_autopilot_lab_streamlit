"""
可視化（Matplotlib / Plotly）

“ゲームの状態”を受け取って、図を作るだけに徹します。
状態遷移やUI（Streamlit）ロジックはここに入れません。
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Polygon, Circle, Ellipse

import plotly.graph_objects as go

from .config import GameConfig
from .model import (
    GameState, Section,
    preview_next, comm_available,
    beta_eff, beta_pointing,
    ephem_at_day, sc_nominal_at_index, sc_nominal_at_fraction,
    earth_angle_base_deg, predicted_earth_angle_deg,
)


# -----------------------------
# ちょっとした共通部品
# -----------------------------
def annotate_outlined(ax, x, y, text, dx=10, dy=10):
    """黒フチ文字（暗背景で読みやすくする）"""
    t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                    color="white", fontsize=10, ha="left", va="bottom")
    t.set_path_effects([pe.Stroke(linewidth=3, foreground="black"), pe.Normal()])
    return t

def controllability_poly(section: Section) -> np.ndarray:
    """Δβの範囲（箱）を、B-plane上の多角形に写像したもの（概念）"""
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

def ellipse_params(cov: np.ndarray, k_sigma: float = 1.0) -> Tuple[float, float, float]:
    """
    2D共分散 → 楕円（幅/高さ/角度）
    ※ “楕円が時々出ない”の主因は、幅・高さがほぼ0になるケースだったので、
       呼び出し側で最小サイズを与える。
    """
    cov = cov + np.eye(2) * 1e-9
    w, V = np.linalg.eigh(cov)
    w = np.maximum(w, 1e-12)
    order = np.argsort(w)[::-1]
    w = w[order]
    V = V[:, order]
    a = k_sigma * math.sqrt(float(w[0]))
    b = k_sigma * math.sqrt(float(w[1]))
    width = 2.0 * a
    height = 2.0 * b
    ang = math.degrees(math.atan2(V[1, 0], V[0, 0]))
    return width, height, ang


# -----------------------------
# B-plane 図（メイン）
# -----------------------------
def plot_bplane(state: GameState, cfg: GameConfig, sections: List[Section], show_truth: bool):
    sec = sections[min(state.k, len(sections) - 1)]
    pv = preview_next(state, cfg, sections, sec)
    B_pred = pv["B_pred"]
    cov_pred = pv["cov_pred"]

    tighten = (state.k + 1) >= cfg.target_tighten_section
    target_r = cfg.target_radius_late_km if tighten else cfg.target_radius_early_km

    poly = controllability_poly(sec) + state.B_est.reshape(1, 2)

    fig = plt.figure(figsize=(11.6, 5.9), dpi=150)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor("#0b0f16")
    ax.set_facecolor("#0b0f16")

    # 制御可能範囲（面）
    ax.add_patch(Polygon(poly, closed=True, facecolor="#00d1ff", edgecolor="none", alpha=0.08, zorder=1))
    ax.plot(poly[:, 0], poly[:, 1], linestyle="--", linewidth=2.0, color="#00d1ff", alpha=0.85,
            label="制御可能範囲（境界）", zorder=2)

    # ターゲット（的）
    ax.add_patch(Circle((cfg.target[0], cfg.target[1]), target_r, fill=False, linewidth=2.6,
                        edgecolor="#ffcc00", alpha=0.95, zorder=3))
    ax.plot([], [], color="#ffcc00", linewidth=2.6, label="ターゲット半径")

    # 予測楕円（1σ）
    w, h, ang = ellipse_params(cov_pred, k_sigma=cfg.pred_ellipse_sigma)

    # “楕円が見えない”対策：最小サイズを入れる
    # （単位はkmなので、数十km以下は見えない＝消えたように見える）
    w = max(float(w), 350.0)
    h = max(float(h), 350.0)

    ax.add_patch(Ellipse((B_pred[0], B_pred[1]), width=w, height=h, angle=ang, fill=False,
                         edgecolor="white", linewidth=2.2, linestyle=":", alpha=0.95, zorder=4))
    ax.plot([], [], color="white", linestyle=":", linewidth=2.2, label="予測範囲（1σ）")

    # 点
    ax.scatter([cfg.target[0]], [cfg.target[1]], s=70, color="#8aa2c8", zorder=5, label="ターゲット中心")
    ax.scatter([state.B_est[0]], [state.B_est[1]], s=90, color="#5dade2", marker="s", zorder=6, label="推定点E（いま）")
    ax.scatter([B_pred[0]], [B_pred[1]], s=90, color="white", marker="o", zorder=7, label="予測中心")

    if state.B_obs_last is not None:
        ax.scatter([state.B_obs_last[0]], [state.B_obs_last[1]], s=95, color="#aab7b8", marker="P",
                   zorder=6, label="観測点（前ターン）")
    if show_truth:
        ax.scatter([state.B_true[0]], [state.B_true[1]], s=95, color="#ff6b6b", marker="^",
                   zorder=7, label="真値（先生）")

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


# -----------------------------
# 2D軌道図（ノミナルを曲線で表示）
# -----------------------------
def plot_orbits_2d_nominal(state: GameState, cfg: GameConfig, sections: List[Section]):
    eph = cfg.eph
    n = len(sections)

    sec_now = sections[min(state.k, n - 1)]
    t_now = sec_now.t_day
    e = ephem_at_day(t_now, eph)
    earth = e["earth"]
    venus = e["venus"]

    # 曲線表示：uを細かく刻んでノミナルをサンプリング
    u_now = 0.0 if n <= 1 else min(1.0, float(state.k) / float(n - 1))
    us = np.linspace(0.0, max(u_now, 1e-6), 200)
    pts_sc = np.stack([sc_nominal_at_fraction(u, eph)["sc"] for u in us], axis=0)

    ths = np.linspace(0, 2 * math.pi, 400)
    earth_orb = np.stack([eph.r_earth * np.cos(ths), eph.r_earth * np.sin(ths)], axis=1)
    venus_orb = np.stack([eph.r_venus * np.cos(ths), eph.r_venus * np.sin(ths)], axis=1)

    fig = plt.figure(figsize=(11.2, 4.4), dpi=150)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor("#0b0f16")
    ax.set_facecolor("#0b0f16")

    ax.plot(earth_orb[:, 0], earth_orb[:, 1], color="#3b82f6", alpha=0.55, linewidth=1.8, label="地球軌道")
    ax.plot(venus_orb[:, 0], venus_orb[:, 1], color="#22c55e", alpha=0.55, linewidth=1.8, label="金星軌道")

    ax.scatter([0], [0], s=140, color="#ffcc00", label="太陽", zorder=6)
    ax.scatter([earth[0]], [earth[1]], s=85, color="#3b82f6", label="地球（いま）", zorder=7)
    ax.scatter([venus[0]], [venus[1]], s=85, color="#22c55e", label="金星（いま）", zorder=7)

    # ノミナル（曲線）
    ax.plot(pts_sc[:, 0], pts_sc[:, 1], color="white", linewidth=2.8, alpha=0.95, label="IKAROS（計画：ノミナル）", zorder=8)
    ax.scatter([pts_sc[-1, 0]], [pts_sc[-1, 1]], color="white", s=90, zorder=9, label="IKAROS（いま）")

    # 到着一致チェック（最終u=1）
    vend = ephem_at_day(eph.t_end_day, eph)["venus"]
    scend = sc_nominal_at_index(n - 1, n, eph)["sc"]
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


# -----------------------------
# βin×βout マップ（電力収支 / DL量）
# -----------------------------
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
            bi = float(X[i, j])
            bo = float(Y[i, j])
            be = beta_eff(bi, bo)
            gen = cfg.gen_scale * max(0.0, math.cos(math.radians(be)))
            ok = comm_available(bi, bo, state, cfg, sections)
            comm_ok[i, j] = 1.0 if ok else 0.0
            cost = cfg.base_load + (cfg.comm_cost if ok else 0.0)
            net[i, j] = gen - cost
            down[i, j] = cfg.data_downlink_cap if ok else 0.0

    return xs, ys, net, down, comm_ok

def plot_beta_maps(state: GameState, cfg: GameConfig, sections: List[Section]):
    xs, ys, net, down, comm_ok = beta_map_grids(state, cfg, sections, step=2.0)

    fig = plt.figure(figsize=(10.8, 4.9), dpi=150)
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

    # 通信OK領域を“緑線＋薄塗り”で見せる
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
# 幾何 3D（Plotly）
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

    # 概念ベクトル：太陽光を +X とする
    sun = np.array([1.0, 0.0, 0.0])
    earth = np.array([math.cos(math.radians(ea)), math.sin(math.radians(ea)), 0.0])

    # 帆法線：βpointでZ回転 → βeffでY回転（概念）
    n = rot_z(bp) @ rot_y(be) @ np.array([1.0, 0.0, 0.0])
    n_norm = n / (np.linalg.norm(n) + 1e-9)

    # 帆平面（法線に直交する四角形）
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
    fig.add_trace(go.Scatter3d(x=corners[:, 0], y=corners[:, 1], z=corners[:, 2], mode="lines",
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

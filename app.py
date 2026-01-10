# IKAROS GO! (Streamlit) - ultra simple version
# ------------------------------------------------------------
# Goal: HTV GO!-level simplicity (one screen, short play, minimal explanation).
# Safe: no exec/eval.
#
# Controls:
# - One slider "ハンドル" (-100..100): bias panels to turn.
# - One button "すすめる": advances time.
#
# Stages (3):
# 1) 太陽を向く（発電ゲージを満たす）
# 2) ゆらゆらセンサーでも安定
# 3) 目的地へ（少しズラして進む）
#
# Teacher mode (optional):
# - shows advanced knobs and logs (collapsed by default)
# ------------------------------------------------------------

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def wrap_pi(a: float) -> float:
    a = (a + math.pi) % (2 * math.pi) - math.pi
    if a <= -math.pi:
        a += 2 * math.pi
    return a


def deg(x: float) -> float:
    return x * 180.0 / math.pi


def rad(x: float) -> float:
    return x * math.pi / 180.0


def vec(theta: float) -> np.ndarray:
    return np.array([math.cos(theta), math.sin(theta)], dtype=float)


def power_from_error(err: float) -> float:
    return clamp(math.cos(err), 0.0, 1.0)


@dataclass
class Params:
    dt: float = 0.1
    mass: float = 1.0
    inertia: float = 0.35
    damping: float = 0.55
    F0: float = 0.17
    T0: float = 0.55


@dataclass
class State:
    t: float
    theta: float
    omega: float
    pos: np.ndarray
    vel: np.ndarray
    score: float
    battery: float  # 0..1
    stable: float   # 0..1
    reached: int
    last_err_sign: int
    flips: int


def init_state() -> State:
    return State(
        t=0.0,
        theta=rad(25.0),
        omega=0.0,
        pos=np.array([0.0, 0.0], dtype=float),
        vel=np.array([0.0, 0.0], dtype=float),
        score=0.0,
        battery=0.0,
        stable=0.0,
        reached=0,
        last_err_sign=0,
        flips=0,
    )


def control_from_handle(handle: float, base: float = 0.65, k: float = 0.35) -> np.ndarray:
    h = clamp(handle / 100.0, -1.0, 1.0)
    L = clamp(base + k * h, 0.0, 1.0)
    R = clamp(base - k * h, 0.0, 1.0)
    Fp = clamp(base * 0.15, 0.0, 1.0)
    B = 0.0
    return np.array([L, R, Fp, B], dtype=float)


def step(state: State, panels: np.ndarray, sun_dir: float, disturbance: float, params: Params) -> Dict[str, float]:
    dt = params.dt
    panels = np.clip(panels, 0.0, 1.0)

    err = wrap_pi(sun_dir - state.theta)

    # sign flips (overshoot)
    sign = 0
    if err > 1e-6:
        sign = 1
    elif err < -1e-6:
        sign = -1
    if state.last_err_sign != 0 and sign != 0 and sign != state.last_err_sign:
        state.flips += 1
    if sign != 0:
        state.last_err_sign = sign

    alpha = err
    mean_r = float(np.mean(panels))

    F = params.F0 * mean_r * max(math.cos(alpha), 0.0)
    L, R, Fp, B = float(panels[0]), float(panels[1]), float(panels[2]), float(panels[3])
    tau = params.T0 * ((R - L) * math.cos(alpha) + (Fp - B) * math.sin(alpha))
    tau += disturbance * (0.6 * math.sin(0.18 * state.t) + 0.4 * math.cos(0.11 * state.t))

    state.omega += (tau / params.inertia - params.damping * state.omega) * dt
    state.theta = wrap_pi(state.theta + state.omega * dt)

    acc = (F / params.mass) * vec(state.theta)
    state.vel = state.vel + acc * dt
    state.pos = state.pos + state.vel * dt

    pwr = power_from_error(err)
    state.score += (8.0 * pwr - 1.0 * abs(err) - 0.25 * abs(state.omega) - 0.25 * float(np.sum(panels))) * dt

    state.t += dt

    return dict(
        t=state.t,
        err=err,
        omega=state.omega,
        power=pwr,
        pos_x=float(state.pos[0]), pos_y=float(state.pos[1]),
        score=float(state.score),
        flips=float(state.flips),
        L=L, R=R, Fp=Fp, B=B,
    )


STAGES = {
    1: dict(
        name="Stage 1：太陽を向け！",
        time_limit=45.0,
        target_battery=1.0,
        noise_deg=0.0,
        disturbance=0.0,
        target=None,
        rules="電力ゲージを満タン（太陽に向くほど増える）",
    ),
    2: dict(
        name="Stage 2：ゆらゆらでも安定",
        time_limit=50.0,
        target_battery=1.0,
        noise_deg=6.0,
        disturbance=0.0,
        target=None,
        rules="安定ゲージを満タン（ズレ小＆回転小で増える）",
    ),
    3: dict(
        name="Stage 3：目的地へ！",
        time_limit=60.0,
        target_battery=0.65,
        noise_deg=3.0,
        disturbance=0.25,
        target=np.array([4.5, 1.2], dtype=float),
        rules="目的地に到達（少しズラして進む）",
    ),
}


def title_rank(score: float) -> str:
    if score >= 260:
        return "伝説の帆船管制官"
    if score >= 210:
        return "エース操縦士"
    if score >= 160:
        return "いい感じの船乗り"
    if score >= 110:
        return "見習い操縦士"
    return "はじめての帆船"


st.set_page_config(page_title="IKAROS GO!", layout="wide")
st.title("☀️ IKAROS GO!（超シンプル版）")
st.caption("1）ハンドルで曲げる　2）太陽マークに向ける　3）ゲージを満たしてクリア！")

colA, colB = st.columns([1.15, 0.85], gap="large")

with st.sidebar:
    st.header("ステージ")
    stage = st.radio("選ぶ", [1, 2, 3], format_func=lambda i: STAGES[i]["name"], index=0)
    st.divider()
    teacher = st.toggle("先生モード（詳細設定）", value=False)

# Session init
if "state" not in st.session_state:
    st.session_state.state = init_state()
    st.session_state.telemetry = []
    st.session_state.sun_dir = 0.0
    st.session_state.params = Params()
    st.session_state.stage = stage

if st.session_state.stage != stage:
    st.session_state.state = init_state()
    st.session_state.telemetry = []
    st.session_state.stage = stage

S = STAGES[stage]
params: Params = st.session_state.params

noise_deg = float(S["noise_deg"])
disturbance = float(S["disturbance"])
time_limit = float(S["time_limit"])

with st.sidebar:
    if teacher:
        st.subheader("難しさ（先生用）")
        noise_deg = st.slider("センサーゆらぎ（度）", 0.0, 12.0, noise_deg, 0.5)
        disturbance = st.slider("外乱（回される）", 0.0, 1.0, disturbance, 0.05)
        time_limit = st.slider("制限時間（秒）", 20.0, 120.0, time_limit, 5.0)

    st.divider()
    st.header("操作")
    handle = st.slider("ハンドル（左← →右）", -100, 100, 0, 1)
    advance = st.radio("すすめる量", ["ちょっと（0.5秒）", "ふつう（1秒）", "まとめて（5秒）"], index=1)
    step_btn = st.button("▶️ すすめる", use_container_width=True)
    reset_btn = st.button("🔁 リセット", use_container_width=True)

if reset_btn:
    st.session_state.state = init_state()
    st.session_state.telemetry = []
    st.rerun()

def advance_steps() -> int:
    if advance.startswith("ちょっと"):
        return int(0.5 / params.dt)
    if advance.startswith("ふつう"):
        return int(1.0 / params.dt)
    return int(5.0 / params.dt)

def sense_err_omega(state: State) -> Tuple[float, float]:
    err = wrap_pi(st.session_state.sun_dir - state.theta)
    e_noisy = rad(deg(err) + np.random.normal(0.0, noise_deg))
    w_noisy = state.omega + rad(np.random.normal(0.0, noise_deg * 0.2))
    return e_noisy, w_noisy

def update_progress(state: State, err_true: float, power: float):
    if stage == 1:
        gain = 0.75 * power
        drain = 0.12 * abs(err_true)
        state.battery = clamp(state.battery + (gain - drain) * params.dt, 0.0, 1.0)
    elif stage == 2:
        ok_err = abs(deg(err_true)) <= 8.0
        ok_w = abs(deg(state.omega)) <= 18.0
        gain = 0.9 if (ok_err and ok_w) else 0.0
        drain = 0.25 if (not ok_err or not ok_w) else 0.0
        state.stable = clamp(state.stable + (gain - drain) * params.dt, 0.0, 1.0)

def check_clear(state: State) -> Tuple[bool, str]:
    if stage == 1:
        if state.battery >= S["target_battery"]:
            return True, "電力ゲージ満タン！太陽に勝った（？）"
        return False, ""
    if stage == 2:
        if state.stable >= 1.0:
            return True, "安定ゲージ満タン！いい操縦〜！"
        return False, ""
    target = S["target"]
    if target is not None:
        if float(np.linalg.norm(state.pos - target)) < 0.35:
            err = wrap_pi(st.session_state.sun_dir - state.theta)
            if power_from_error(err) >= S["target_battery"]:
                state.reached = 1
                return True, "目的地到達！しかも発電もキープ！"
            return False, "着いたけど…太陽から背を向けすぎ！ちょい修正。"
    return False, ""

if step_btn:
    n = advance_steps()
    for _ in range(n):
        if st.session_state.state.t >= time_limit:
            break

        state: State = st.session_state.state
        e_noisy, w_noisy = sense_err_omega(state)  # reserved for future, logged in teacher mode

        panels = control_from_handle(handle)
        telem = step(state, panels, st.session_state.sun_dir, disturbance, params)

        update_progress(state, telem["err"], telem["power"])

        if teacher:
            telem["e_noisy_deg"] = deg(e_noisy)
            telem["w_noisy_deg_s"] = deg(w_noisy)
        st.session_state.telemetry.append(telem)

        cleared, _ = check_clear(state)
        if cleared:
            break

state: State = st.session_state.state
err_true = wrap_pi(st.session_state.sun_dir - state.theta)
pwr = power_from_error(err_true)

with colA:
    st.subheader(S["rules"])

    g1, g2, g3 = st.columns([1, 1, 1])
    g1.metric("時間", f"{state.t:.1f}/{time_limit:.0f} s")
    g2.metric("太陽ズレ", f"{deg(err_true):.1f}°")
    g3.metric("電力", f"{pwr:.2f}")

    if stage == 1:
        st.progress(state.battery, text=f"電力ゲージ：{int(state.battery*100)}%")
    elif stage == 2:
        st.progress(state.stable, text=f"安定ゲージ：{int(state.stable*100)}%")
    else:
        tgt = S["target"]
        dist = float(np.linalg.norm(state.pos - tgt)) if tgt is not None else 0.0
        st.progress(clamp(1.0 - dist / 5.0, 0.0, 1.0), text=f"目的地まで：{dist:.2f}")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if st.session_state.telemetry:
        xs = [t["pos_x"] for t in st.session_state.telemetry]
        ys = [t["pos_y"] for t in st.session_state.telemetry]
        ax.plot(xs, ys, linewidth=2)

    p = state.pos
    fwd = vec(state.theta)
    ax.arrow(p[0], p[1], 0.35 * fwd[0], 0.35 * fwd[1], head_width=0.12, length_includes_head=True)

    sun_vec = vec(st.session_state.sun_dir)
    ax.arrow(p[0], p[1], 0.45 * sun_vec[0], 0.45 * sun_vec[1], head_width=0.10, length_includes_head=True)

    if stage == 3 and S["target"] is not None:
        tgt = S["target"]
        ax.scatter([tgt[0]], [tgt[1]], marker="o")
        ax.text(tgt[0] + 0.05, tgt[1] + 0.05, "GOAL", fontsize=11)

    if st.session_state.telemetry:
        xs = np.array([t["pos_x"] for t in st.session_state.telemetry] + [p[0]])
        ys = np.array([t["pos_y"] for t in st.session_state.telemetry] + [p[1]])
        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())
        pad = 0.9
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)
    else:
        ax.set_xlim(-1.5, 3.5)
        ax.set_ylim(-1.5, 2.5)

    st.pyplot(fig, use_container_width=True)

    cleared, msg = check_clear(state)
    if cleared:
        st.success(msg)
        st.balloons()
        st.markdown(f"**称号：{title_rank(state.score)}**")
    elif state.t >= time_limit:
        st.error("時間切れ！もう一回！")
        st.markdown(f"称号：{title_rank(state.score)}")

with colB:
    st.subheader("遊び方（これだけ）")
    st.markdown(
        """
- **ハンドル**を動かすと、帆が左右に“じわっ”と回る  
- **太陽マーク（矢印）**の方向に向けると電力が増える  
- Stage3は **少しズラして進む**（でもズラしすぎると電力が落ちる）
"""
    )

    st.divider()
    st.markdown("**いまのコツ**")
    if stage == 1:
        st.info("ズレが小さくなったらハンドルを0へ。チョン操作が強い。")
    elif stage == 2:
        st.info("ノイズでフラつく。反応しすぎず、ゆっくり戻す。")
    else:
        st.info("目的地へ向けて少しズラす。でも電力0.65未満だと“失速”。")

    if teacher:
        st.divider()
        st.subheader("ログ（先生モード）")
        if st.session_state.telemetry:
            with st.expander("角度ズレ・角速度（グラフ）", expanded=False):
                t = np.array([x["t"] for x in st.session_state.telemetry], dtype=float)
                e = np.degrees(np.array([x["err"] for x in st.session_state.telemetry], dtype=float))
                w = np.degrees(np.array([x["omega"] for x in st.session_state.telemetry], dtype=float))

                fig1 = plt.figure()
                ax1 = fig1.add_subplot(111)
                ax1.set_xlabel("t (s)")
                ax1.set_ylabel("error (deg)")
                ax1.plot(t, e)
                st.pyplot(fig1, use_container_width=True)

                fig2 = plt.figure()
                ax2 = fig2.add_subplot(111)
                ax2.set_xlabel("t (s)")
                ax2.set_ylabel("omega (deg/s)")
                ax2.plot(t, w)
                st.pyplot(fig2, use_container_width=True)

            with st.expander("表（最後の100行）", expanded=False):
                st.dataframe(st.session_state.telemetry[-100:], use_container_width=True)
        else:
            st.caption("まだログがありません。")

st.caption("※ これは教材用の簡略モデルです（実機の正確な物理モデルではありません）。")

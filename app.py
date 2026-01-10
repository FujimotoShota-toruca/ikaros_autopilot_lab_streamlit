# IKAROS Autopilot Lab (Streamlit)
# ------------------------------------------------------------
# A kid-friendly, single-file web app for learning feedback control
# using an IKAROS-inspired light-sail "4-panel" control model.
#
# Notes
# - Runs well on Streamlit Community Cloud (GitHub -> Deploy).
# - Does NOT execute arbitrary user Python (security). Instead, it
#   offers safe modes: Manual / Rule / PD / MiniScript (parsed).
#
# Author: (your team / 宇宙少年団)
# License: MIT (you can change)
# ------------------------------------------------------------

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def wrap_pi(a: float) -> float:
    """Wrap angle to (-pi, pi]."""
    a = (a + math.pi) % (2 * math.pi) - math.pi
    # avoid -pi (pure aesthetics)
    if a <= -math.pi:
        a += 2 * math.pi
    return a

def deg(x_rad: float) -> float:
    return x_rad * 180.0 / math.pi

def rad(x_deg: float) -> float:
    return x_deg * math.pi / 180.0

def vec_from_angle(theta: float) -> np.ndarray:
    return np.array([math.cos(theta), math.sin(theta)], dtype=float)


# -----------------------------
# Mission definitions (10 cards)
# -----------------------------
@dataclass
class Mission:
    key: str
    title: str
    level: str
    goal: str
    clear: str
    hint: str
    env: Dict[str, float]  # noise_deg, delay_steps, disturbance, time_limit_s, checkpoints (0/1)
    scoring: Dict[str, float]  # weights

def missions() -> List[Mission]:
    # Default scoring weights (tweakable)
    base_scoring = dict(
        w_power=8.0,        # per second * power
        w_error=1.0,        # per second * |error_rad|
        w_control=0.6,      # per second * sum(panel)
        w_spin=0.25,        # per second * |omega|
        bonus_checkpoint=80.0,
        penalty_flip=3.0,   # per sign-flip of error
    )
    return [
        Mission(
            key="M01",
            title="太陽に顔を向けろ！",
            level="Lv0",
            goal="60秒間、太陽角（ズレ）をできるだけ小さく保つ",
            clear="残り時間0で生存（電力が0になり続けない）",
            hint="ズレが大きい時だけ動かすと省エネ。",
            env=dict(noise_deg=0.0, delay_steps=0, disturbance=0.0, time_limit_s=60.0, checkpoints=0),
            scoring=base_scoring,
        ),
        Mission(
            key="M02",
            title="ぴた止め職人",
            level="Lv0",
            goal="行き過ぎ（ズレの符号反転）を減らす",
            clear="60秒で行き過ぎ回数が少ないほど高得点",
            hint="近づいたら弱める（チョン押し）がコツ。",
            env=dict(noise_deg=0.0, delay_steps=0, disturbance=0.0, time_limit_s=60.0, checkpoints=0),
            scoring={**base_scoring, "penalty_flip": 6.0},
        ),
        Mission(
            key="M03",
            title="もし〜なら操縦",
            level="Lv1",
            goal="自動操縦で60秒生存",
            clear="オートONで最後まで生存",
            hint="e>+しきい値 と e<-しきい値 の2条件から始める。",
            env=dict(noise_deg=0.0, delay_steps=0, disturbance=0.0, time_limit_s=60.0, checkpoints=0),
            scoring=base_scoring,
        ),
        Mission(
            key="M04",
            title="ノイズ警報！ゆらゆらセンサー",
            level="Lv1",
            goal="センサーがブレても安定",
            clear="ノイズONで60秒生存＋スコア規定以上を狙う",
            hint="小さいズレに反応しすぎない（無視ゾーン）。",
            env=dict(noise_deg=6.0, delay_steps=0, disturbance=0.0, time_limit_s=60.0, checkpoints=0),
            scoring=base_scoring,
        ),
        Mission(
            key="M05",
            title="電力ケチケチ王",
            level="Lv1",
            goal="生存しつつ操作量を減らす",
            clear="60秒生存＋操作量を抑えて高得点",
            hint="必要な時だけ動かす／全部OFFの時間を作る。",
            env=dict(noise_deg=3.0, delay_steps=0, disturbance=0.0, time_limit_s=60.0, checkpoints=0),
            scoring={**base_scoring, "w_control": 1.2},
        ),
        Mission(
            key="M06",
            title="PD入門：ぴたっと止めろ",
            level="Lv2",
            goal="太陽角±5°以内の滞在時間を最大化",
            clear="滞在時間を伸ばす（行き過ぎ少なく）",
            hint="eだけだと行き過ぎる。ωが大きい時は弱める。",
            env=dict(noise_deg=2.0, delay_steps=0, disturbance=0.0, time_limit_s=60.0, checkpoints=0),
            scoring=base_scoring,
        ),
        Mission(
            key="M07",
            title="操作に遅れあり！ラグ操縦",
            level="Lv2",
            goal="遅延ONでも安定させる",
            clear="遅延ONで60秒生存＋スコア規定以上",
            hint="強すぎると遅れで暴れる。Kp下げる or Kd上げる。",
            env=dict(noise_deg=2.0, delay_steps=2, disturbance=0.0, time_limit_s=60.0, checkpoints=0),
            scoring=base_scoring,
        ),
        Mission(
            key="M08",
            title="外乱だ！宇宙のゆびで押された",
            level="Lv2",
            goal="外乱トルクに負けない",
            clear="外乱ONで60秒生存",
            hint="外乱があると、無視ゾーンを広げすぎると追従できない。",
            env=dict(noise_deg=2.0, delay_steps=1, disturbance=0.35, time_limit_s=60.0, checkpoints=0),
            scoring=base_scoring,
        ),
        Mission(
            key="M09",
            title="チェックポイント航海",
            level="Lv3",
            goal="チェックポイントを順番に3つ通過",
            clear="制限時間内に3点通過",
            hint="太陽に完全一致だと進みたい方向が作れない。少しズラして推進。",
            env=dict(noise_deg=2.0, delay_steps=1, disturbance=0.2, time_limit_s=75.0, checkpoints=1),
            scoring={**base_scoring, "bonus_checkpoint": 120.0},
        ),
        Mission(
            key="M10",
            title="最適化バトル：点取り職人",
            level="Lv3",
            goal="総合スコア最大化（発電・到達・省エネ・安定）",
            clear="ランキング勝負（同点なら操作量が少ない方）",
            hint="速さと省エネはトレードオフ。評価関数で勝つ。",
            env=dict(noise_deg=3.0, delay_steps=2, disturbance=0.25, time_limit_s=90.0, checkpoints=1),
            scoring={**base_scoring, "w_control": 0.85, "w_error": 1.1, "bonus_checkpoint": 130.0},
        ),
    ]


# -----------------------------
# Simulation model (intentionally simple)
# -----------------------------
@dataclass
class SimParams:
    dt: float = 0.1
    mass: float = 1.0
    inertia: float = 0.35
    damping: float = 0.55
    F0: float = 0.16     # "base thrust" (game units)
    T0: float = 0.45     # "base torque" (game units)

@dataclass
class SimState:
    t: float
    theta: float
    omega: float
    pos: np.ndarray
    vel: np.ndarray
    score: float
    flips: int
    last_err_sign: int
    checkpoints: List[np.ndarray]
    passed: int
    control_queue: List[np.ndarray]  # for delay

def default_checkpoints() -> List[np.ndarray]:
    return [np.array([2.5, 1.2]), np.array([4.5, -0.6]), np.array([6.5, 1.0])]

def init_state(delay_steps: int) -> SimState:
    q = [np.zeros(4, dtype=float) for _ in range(max(0, delay_steps))]
    return SimState(
        t=0.0,
        theta=rad(20.0),
        omega=0.0,
        pos=np.array([0.0, 0.0], dtype=float),
        vel=np.array([0.0, 0.0], dtype=float),
        score=0.0,
        flips=0,
        last_err_sign=0,
        checkpoints=default_checkpoints(),
        passed=0,
        control_queue=q,
    )

def power_from_error(err: float) -> float:
    # simple: max when facing sun
    return clamp(math.cos(err), 0.0, 1.0)

def apply_control_delay(u: np.ndarray, queue: List[np.ndarray]) -> np.ndarray:
    if not queue:
        return u
    queue.append(u.copy())
    return queue.pop(0)

def step_sim(
    stt: SimState,
    u_panels: np.ndarray,
    sun_dir: float,
    env_noise_deg: float,
    env_disturbance: float,
    scoring: Dict[str, float],
    params: SimParams,
    delay_steps: int,
    enable_checkpoints: bool,
) -> Tuple[SimState, Dict[str, float]]:
    dt = params.dt

    # Sensing (for score, show both true & noisy later)
    err_true = wrap_pi(sun_dir - stt.theta)
    omega_true = stt.omega

    # count sign flips (overshoot proxy)
    sign = 0
    if err_true > 1e-6:
        sign = 1
    elif err_true < -1e-6:
        sign = -1
    if stt.last_err_sign != 0 and sign != 0 and sign != stt.last_err_sign:
        stt.flips += 1
        stt.score -= scoring.get("penalty_flip", 0.0)
    if sign != 0:
        stt.last_err_sign = sign

    # delay and saturation
    u_panels = np.clip(u_panels, 0.0, 1.0)
    u_applied = apply_control_delay(u_panels, stt.control_queue if delay_steps > 0 else [])

    # dynamics
    alpha = err_true  # relative sun angle
    # Thrust along ship forward; scaled by mean reflectivity and alignment
    F = params.F0 * float(np.mean(u_applied)) * max(math.cos(alpha), 0.0)
    # Torque from asymmetry. Uses both L/R and F/B with different phase, for "4 panels" feeling.
    L, R, Fp, B = float(u_applied[0]), float(u_applied[1]), float(u_applied[2]), float(u_applied[3])
    tau = params.T0 * ((R - L) * math.cos(alpha) + (Fp - B) * math.sin(alpha))

    # add low-frequency disturbance torque (random-walk-ish)
    # deterministic-ish per run: use sin/cos of time; plus optional small noise
    tau += env_disturbance * (0.6 * math.sin(0.18 * stt.t) + 0.4 * math.cos(0.11 * stt.t))

    # integrate rotation
    domega = (tau / params.inertia - params.damping * stt.omega) * dt
    stt.omega += domega
    stt.theta = wrap_pi(stt.theta + stt.omega * dt)

    # integrate translation
    acc = (F / params.mass) * vec_from_angle(stt.theta)
    stt.vel = stt.vel + acc * dt
    stt.pos = stt.pos + stt.vel * dt

    # score
    pwr = power_from_error(err_true)
    stt.score += scoring.get("w_power", 0.0) * pwr * dt
    stt.score -= scoring.get("w_error", 0.0) * abs(err_true) * dt
    stt.score -= scoring.get("w_control", 0.0) * float(np.sum(u_applied)) * dt
    stt.score -= scoring.get("w_spin", 0.0) * abs(stt.omega) * dt

    # checkpoint handling
    bonus = 0.0
    if enable_checkpoints and stt.passed < len(stt.checkpoints):
        cp = stt.checkpoints[stt.passed]
        if float(np.linalg.norm(stt.pos - cp)) < 0.35:
            stt.passed += 1
            bonus = scoring.get("bonus_checkpoint", 0.0)
            stt.score += bonus

    # advance time
    stt.t += dt

    # return telemetry
    telemetry = dict(
        t=stt.t,
        err_true=err_true,
        omega=omega_true,
        power=pwr,
        F=F,
        tau=tau,
        L=L, R=R, Fp=Fp, B=B,
        pos_x=float(stt.pos[0]), pos_y=float(stt.pos[1]),
        score=float(stt.score),
        flips=float(stt.flips),
        checkpoint_bonus=float(bonus),
        passed=float(stt.passed),
    )
    return stt, telemetry


# -----------------------------
# Control modes (safe)
# -----------------------------
def manual_control(ui: Dict[str, float]) -> np.ndarray:
    return np.array([ui["L"], ui["R"], ui["F"], ui["B"]], dtype=float)

def rule_control(err_noisy_deg: float, ui: Dict[str, float]) -> np.ndarray:
    thr1 = ui["thr1"]
    thr2 = ui["thr2"]
    levels = int(ui["levels"])  # 2 or 3
    deadband = ui["deadband"]
    # default all off
    L = R = Fp = B = 0.0

    e = err_noisy_deg
    if abs(e) <= deadband:
        return np.zeros(4, dtype=float)

    def strength(a: float) -> float:
        if levels <= 2:
            return 1.0
        # 3-level
        if abs(a) >= thr2:
            return 1.0
        return 0.6

    s = strength(e)
    # simple: if sun is to the "left" (positive error), torque right by brightening left side
    if e > thr1:
        L, R = s, 0.0
    elif e < -thr1:
        R, L = s, 0.0
    else:
        # within thr1 but outside deadband: small nudge
        if e > 0:
            L, R = 0.35, 0.0
        else:
            R, L = 0.35, 0.0

    # optional "forward/back" micro adjustments (for flavor)
    # if pointing far away, increase forward panel to get extra torque component
    if abs(e) > 25:
        Fp = 0.25
    return np.array([L, R, Fp, B], dtype=float)

def pd_control(err_noisy_rad: float, omega_noisy: float, ui: Dict[str, float]) -> np.ndarray:
    Kp = ui["Kp"]
    Kd = ui["Kd"]
    deadband_deg = ui["deadband"]
    maxu = ui["maxu"]
    # deadband in rad
    if abs(deg(err_noisy_rad)) <= deadband_deg:
        return np.zeros(4, dtype=float)

    u = Kp * err_noisy_rad - Kd * omega_noisy
    u = clamp(u, -maxu, maxu)

    # Map u to panels (L/R mainly); add tiny F/B to keep "4 panels" relevant.
    L = R = Fp = B = 0.0
    if u > 0:
        L = clamp(abs(u) / maxu, 0.0, 1.0)
    elif u < 0:
        R = clamp(abs(u) / maxu, 0.0, 1.0)

    # If error large, bias F panel slightly for more authority
    if abs(deg(err_noisy_rad)) > 20:
        Fp = 0.2 * clamp(abs(u) / maxu, 0.0, 1.0)

    return np.array([L, R, Fp, B], dtype=float)


# MiniScript: parse a tiny rule language (no exec/eval)
SCRIPT_HELP = """\
MiniScript (安全な簡易言語) 例：

# e は「太陽角ズレ（度）」です。正なら左側が明るい方向。
IF e > 12: L=1, R=0, F=0.2, B=0
IF e < -12: R=1, L=0, F=0.2
ELSE: L=0, R=0, F=0, B=0

- 使える変数：e（度）, w（角速度：deg/s）
- 代入できる：L, R, F, B（0〜1）
- 上から順に評価して、最初に当たった IF を採用します。
"""

_rule_re = re.compile(r"^(IF)\s+(.+?)\s*:\s*(.+)$", re.IGNORECASE)
_else_re = re.compile(r"^(ELSE)\s*:\s*(.+)$", re.IGNORECASE)

def _parse_assignments(s: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            raise ValueError(f"代入が読めません: {p}")
        k, v = [x.strip() for x in p.split("=", 1)]
        k = k.upper()
        if k not in ("L", "R", "F", "B"):
            raise ValueError(f"未知の出力 {k}（L,R,F,Bのみ）")
        val = float(v)
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"{k} は 0〜1 で指定してください（{val}）")
        out[k] = val
    return out

def miniscript_control(script: str, e_deg: float, w_deg_s: float) -> np.ndarray:
    # Defaults
    default = np.zeros(4, dtype=float)
    lines = []
    for raw in script.splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        lines.append(raw)

    chosen: Optional[Dict[str, float]] = None
    else_assign: Optional[Dict[str, float]] = None

    for line in lines:
        m_if = _rule_re.match(line)
        m_else = _else_re.match(line)
        if m_if:
            cond = m_if.group(2).strip()
            assigns = _parse_assignments(m_if.group(3).strip())
            # Evaluate condition safely: allow comparisons against e/w with numbers.
            # Supported patterns:
            #   e > 10
            #   e < -12.5
            #   abs(e) > 5
            #   w > 20
            ok = _eval_condition(cond, e_deg, w_deg_s)
            if ok and chosen is None:
                chosen = assigns
        elif m_else:
            else_assign = _parse_assignments(m_else.group(2).strip())
        else:
            raise ValueError(f"行が読めません: {line}")

    if chosen is None and else_assign is not None:
        chosen = else_assign
    if chosen is None:
        return default

    L = chosen.get("L", 0.0)
    R = chosen.get("R", 0.0)
    Fp = chosen.get("F", 0.0)
    B = chosen.get("B", 0.0)
    return np.array([L, R, Fp, B], dtype=float)

_cond_simple = re.compile(r"^(e|w)\s*(<=|>=|<|>|==)\s*(-?\d+(\.\d+)?)$", re.IGNORECASE)
_cond_abs = re.compile(r"^abs\(\s*(e|w)\s*\)\s*(<=|>=|<|>|==)\s*(-?\d+(\.\d+)?)$", re.IGNORECASE)

def _eval_condition(cond: str, e_deg: float, w_deg_s: float) -> bool:
    c = cond.strip().replace(" ", "")
    m = _cond_abs.match(c)
    if m:
        var = m.group(1).lower()
        op = m.group(2)
        val = float(m.group(3))
        x = abs(e_deg) if var == "e" else abs(w_deg_s)
        return _cmp(x, op, val)

    m = _cond_simple.match(c)
    if m:
        var = m.group(1).lower()
        op = m.group(2)
        val = float(m.group(3))
        x = e_deg if var == "e" else w_deg_s
        return _cmp(x, op, val)

    raise ValueError(f"条件が読めません: {cond}\n対応: e>10 / abs(e)>5 / w<-20 など")

def _cmp(x: float, op: str, y: float) -> bool:
    if op == "<":
        return x < y
    if op == "<=":
        return x <= y
    if op == ">":
        return x > y
    if op == ">=":
        return x >= y
    if op == "==":
        return x == y
    raise ValueError(f"未知の比較演算子: {op}")


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="IKAROS Autopilot Lab", layout="wide")

st.title("☀️ IKAROS Autopilot Lab（試作）")
st.caption("太陽光で進む“帆船”を、4枚パネルで操縦しよう。低学年は手動で遊べて、高学年は制御で勝てる。")

ms = missions()
mission_map = {m.key: m for m in ms}

# Sidebar
with st.sidebar:
    st.header("設定")

    m_key = st.selectbox(
        "ミッション",
        options=[m.key for m in ms],
        format_func=lambda k: f"{k} {mission_map[k].level}｜{mission_map[k].title}",
        index=0,
    )
    m = mission_map[m_key]

    st.markdown(f"**目的**：{m.goal}")
    st.markdown(f"**クリア**：{m.clear}")
    st.markdown(f"**ヒント**：{m.hint}")

    st.divider()
    mode = st.radio("操作モード", ["手動", "ルール（もし〜なら）", "PD（数式）", "MiniScript"], index=0)

    # Environment (mission default + optional override)
    st.subheader("環境（難しさ）")
    noise_deg = st.slider("センサーゆらぎ（度）", 0.0, 12.0, float(m.env["noise_deg"]), 0.5)
    delay_steps = st.slider("操作の遅れ（フレーム）", 0, 4, int(m.env["delay_steps"]), 1)
    disturbance = st.slider("外乱（回される強さ）", 0.0, 1.0, float(m.env["disturbance"]), 0.05)

    time_limit_s = st.slider("制限時間（秒）", 20.0, 120.0, float(m.env["time_limit_s"]), 5.0)

    enable_checkpoints = st.checkbox("チェックポイントを有効", value=bool(m.env["checkpoints"]))

    st.divider()
    st.subheader("操作（モード別）")

    ui: Dict[str, float] = {}
    if mode == "手動":
        ui["L"] = 1.0 if st.toggle("左パネル ON", value=False) else 0.0
        ui["R"] = 1.0 if st.toggle("右パネル ON", value=False) else 0.0
        ui["F"] = 1.0 if st.toggle("前パネル ON", value=False) else 0.0
        ui["B"] = 1.0 if st.toggle("後パネル ON", value=False) else 0.0

    elif mode == "ルール（もし〜なら）":
        ui["levels"] = float(st.radio("強さ段階", ["2段階（ON/OFF）", "3段階（弱/中/強）"], index=1).startswith("3") and 3 or 2)
        ui["deadband"] = st.slider("無視ゾーン（度）", 0.0, 15.0, 5.0, 0.5)
        ui["thr1"] = st.slider("しきい値1（度）", 1.0, 30.0, 10.0, 0.5)
        ui["thr2"] = st.slider("しきい値2（度）", 5.0, 45.0, 25.0, 0.5)

    elif mode == "PD（数式）":
        ui["Kp"] = st.slider("Kp（角度に反応）", 0.0, 2.0, 0.8, 0.05)
        ui["Kd"] = st.slider("Kd（回転を止める）", 0.0, 2.0, 0.4, 0.05)
        ui["deadband"] = st.slider("無視ゾーン（度）", 0.0, 15.0, 3.0, 0.5)
        ui["maxu"] = st.slider("最大操作（飽和）", 0.2, 2.0, 1.0, 0.05)

    else:
        st.caption("MiniScriptは安全な“簡易言語”です（Pythonは実行しません）。")
        st.code(SCRIPT_HELP, language="text")
        default_script = "IF e > 12: L=1, R=0, F=0.2\nIF e < -12: R=1, L=0, F=0.2\nELSE: L=0, R=0, F=0, B=0\n"
        script = st.text_area("MiniScript", value=default_script, height=180)
        ui["script"] = script

    st.divider()
    st.subheader("実行")
    run_steps = st.slider("Runで進めるステップ数", 10, 600, 120, 10)
    do_step = st.button("▶️ Step（1ステップ）", use_container_width=True)
    do_run = st.button("⏩ Run（まとめて）", use_container_width=True)
    do_reset = st.button("🔁 Reset", use_container_width=True)


# Session state init
if "sim" not in st.session_state:
    st.session_state.sim = init_state(delay_steps=0)
    st.session_state.telemetry = []
    st.session_state.sun_dir = 0.0  # fixed sun direction
    st.session_state.params = SimParams()
    st.session_state.last_mode = mode
    st.session_state.last_delay = delay_steps

def hard_reset():
    st.session_state.sim = init_state(delay_steps=delay_steps)
    st.session_state.telemetry = []
    st.session_state.sun_dir = 0.0
    st.session_state.last_mode = mode
    st.session_state.last_delay = delay_steps

# Reset if delay changed (queue length matters)
if st.session_state.get("last_delay", delay_steps) != delay_steps:
    hard_reset()

if do_reset:
    hard_reset()

# Compute one control command from current mode + noisy measurements
def compute_u(sim: SimState) -> Tuple[np.ndarray, Dict[str, float]]:
    # True values
    err_true = wrap_pi(st.session_state.sun_dir - sim.theta)
    omega_true = sim.omega

    # Noisy sensor (for control input)
    e_noisy_deg = deg(err_true) + np.random.normal(0.0, noise_deg)
    w_noisy_deg_s = deg(omega_true) + np.random.normal(0.0, noise_deg * 0.2)

    info = dict(err_true=err_true, e_noisy_deg=e_noisy_deg, w_noisy_deg_s=w_noisy_deg_s)

    if mode == "手動":
        u = manual_control(ui)
    elif mode == "ルール（もし〜なら）":
        u = rule_control(e_noisy_deg, ui)
    elif mode == "PD（数式）":
        u = pd_control(rad(e_noisy_deg), rad(w_noisy_deg_s), ui)
    else:
        try:
            u = miniscript_control(ui["script"], e_noisy_deg, w_noisy_deg_s)
        except Exception as ex:
            st.warning(f"MiniScriptエラー：{ex}")
            u = np.zeros(4, dtype=float)

    return u, info


# Simulate step(s)
def run_n(n: int):
    sim: SimState = st.session_state.sim
    for _ in range(n):
        if sim.t >= time_limit_s:
            break

        u, sense = compute_u(sim)
        sim, telem = step_sim(
            sim,
            u_panels=u,
            sun_dir=st.session_state.sun_dir,
            env_noise_deg=noise_deg,
            env_disturbance=disturbance,
            scoring=m.scoring,
            params=st.session_state.params,
            delay_steps=delay_steps,
            enable_checkpoints=enable_checkpoints,
        )
        # merge telemetry
        telem["u_L"] = float(u[0]); telem["u_R"] = float(u[1]); telem["u_F"] = float(u[2]); telem["u_B"] = float(u[3])
        telem["e_noisy_deg"] = float(sense["e_noisy_deg"])
        telem["w_noisy_deg_s"] = float(sense["w_noisy_deg_s"])
        st.session_state.telemetry.append(telem)

    st.session_state.sim = sim


if do_step:
    run_n(1)
if do_run:
    run_n(run_steps)


# -----------------------------
# Layout: main view
# -----------------------------
left, right = st.columns([1.15, 1.0], gap="large")

sim: SimState = st.session_state.sim
err_true = wrap_pi(st.session_state.sun_dir - sim.theta)
pwr = power_from_error(err_true)

with left:
    st.subheader("シミュレーション")
    # Plot trajectory and current pose
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if st.session_state.telemetry:
        xs = [t["pos_x"] for t in st.session_state.telemetry]
        ys = [t["pos_y"] for t in st.session_state.telemetry]
        ax.plot(xs, ys, linewidth=2)

    # checkpoints
    if enable_checkpoints:
        cps = sim.checkpoints
        for i, cp in enumerate(cps):
            ax.scatter([cp[0]], [cp[1]], marker="o")
            ax.text(cp[0] + 0.05, cp[1] + 0.05, f"CP{i+1}", fontsize=10)

    # ship arrow
    p = sim.pos
    fwd = vec_from_angle(sim.theta)
    ax.arrow(p[0], p[1], 0.35 * fwd[0], 0.35 * fwd[1], head_width=0.12, length_includes_head=True)

    # sun arrow (from ship pointing to sun direction reference)
    sun_vec = vec_from_angle(st.session_state.sun_dir)
    ax.arrow(p[0], p[1], 0.45 * sun_vec[0], 0.45 * sun_vec[1], head_width=0.10, length_includes_head=True)

    # auto-scale with margins
    if st.session_state.telemetry:
        xs = np.array([t["pos_x"] for t in st.session_state.telemetry] + [p[0]])
        ys = np.array([t["pos_y"] for t in st.session_state.telemetry] + [p[1]])
        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())
        pad = 0.8
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)
    else:
        ax.set_xlim(-1.5, 3.0)
        ax.set_ylim(-1.5, 2.0)

    st.pyplot(fig, use_container_width=True)

    # Quick status
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("時間", f"{sim.t:.1f} s")
    c2.metric("太陽角ズレ", f"{deg(err_true):.1f}°")
    c3.metric("角速度", f"{deg(sim.omega):.1f} °/s")
    c4.metric("電力", f"{pwr:.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("スコア", f"{sim.score:.1f}")
    c6.metric("行き過ぎ回数", f"{sim.flips}")
    if enable_checkpoints:
        c7.metric("CP通過", f"{sim.passed}/{len(sim.checkpoints)}")
    else:
        c7.metric("CP通過", "-")
    c8.metric("モード", mode)


with right:
    st.subheader("学びの表示（ログ）")
    st.caption("高学年向け：『何が起きたか』を見える化します。")

    if not st.session_state.telemetry:
        st.info("まだログがありません。Step か Run を押してね。")
    else:
        t = np.array([x["t"] for x in st.session_state.telemetry], dtype=float)
        e = np.array([x["err_true"] for x in st.session_state.telemetry], dtype=float)
        w = np.array([x["omega"] for x in st.session_state.telemetry], dtype=float)
        u_sum = np.array([x["u_L"] + x["u_R"] + x["u_F"] + x["u_B"] for x in st.session_state.telemetry], dtype=float)
        score = np.array([x["score"] for x in st.session_state.telemetry], dtype=float)

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.set_xlabel("t (s)")
        ax2.set_ylabel("angle error (deg)")
        ax2.plot(t, np.degrees(e))
        st.pyplot(fig2, use_container_width=True)

        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        ax3.set_xlabel("t (s)")
        ax3.set_ylabel("omega (deg/s)")
        ax3.plot(t, np.degrees(w))
        st.pyplot(fig3, use_container_width=True)

        fig4 = plt.figure()
        ax4 = fig4.add_subplot(111)
        ax4.set_xlabel("t (s)")
        ax4.set_ylabel("control sum (0..4)")
        ax4.plot(t, u_sum)
        st.pyplot(fig4, use_container_width=True)

        fig5 = plt.figure()
        ax5 = fig5.add_subplot(111)
        ax5.set_xlabel("t (s)")
        ax5.set_ylabel("score")
        ax5.plot(t, score)
        st.pyplot(fig5, use_container_width=True)

        with st.expander("ログ（CSVっぽく見る）"):
            st.dataframe(st.session_state.telemetry[-200:], use_container_width=True)


st.divider()
st.subheader("先生・運営向けメモ（このアプリの意図）")
st.markdown(
    """
- **低学年**：手動で「ズレ→直す→良くなる」を遊びで体験  
- **高学年**：ルール／PDで“安定化”を作り、ログで原因を考える  
- **中学生**：Lv3で「目的関数のバランス（速さ vs 省エネ）」に沼る  
"""
)

st.caption("※ 本モデルは“教材用の簡略化”です（実機の正確な物理モデルではありません）。")

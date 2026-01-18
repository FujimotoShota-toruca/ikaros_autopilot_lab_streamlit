\
from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = Path(__file__).parent / "data"

# -----------------------------
# 設定（デフォルト）
# -----------------------------
@dataclass
class GameConfig:
    n_turns: int = 14
    turn_days: int = 14

    target_bt_br_km: Tuple[float, float] = (0.0, 0.0)
    tolerance_km: float = 30.0

    beta_max_deg: float = 15.0
    beta_step_deg: float = 0.5

    process_noise_km: float = 3.0
    meas_noise_km: float = 6.0
    init_est_noise_km: float = 10.0

    sun_tilt_limit_deg: float = 45.0
    comm_ok_low_deg: float = 60.0
    comm_ok_high_deg: float = 120.0

    blackout_start_day: float = 100.0
    blackout_end_day: float = 130.0

    init_bt_br_km: Tuple[float, float] = (120.0, -80.0)

CFG = GameConfig()

def load_json(path: Path) -> Optional[object]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

MISSION = load_json(DATA_DIR / "mission_config.json")
if isinstance(MISSION, dict):
    try:
        CFG = replace(
            CFG,
            n_turns=int(MISSION.get("n_turns", CFG.n_turns)),
            turn_days=int(MISSION.get("turn_days", CFG.turn_days)),
            tolerance_km=float(MISSION.get("tolerance_km", CFG.tolerance_km)),
            sun_tilt_limit_deg=float(MISSION.get("sun_tilt_limit_deg", CFG.sun_tilt_limit_deg)),
            blackout_start_day=float(MISSION.get("blackout_start_day", CFG.blackout_start_day)),
            blackout_end_day=float(MISSION.get("blackout_end_day", CFG.blackout_end_day)),
        )
        tb = MISSION.get("target_bt_br_km", None)
        if isinstance(tb, list) and len(tb) == 2:
            CFG = replace(CFG, target_bt_br_km=(float(tb[0]), float(tb[1])))
        ib = MISSION.get("init_bt_br_km", None)
        if isinstance(ib, list) and len(ib) == 2:
            CFG = replace(CFG, init_bt_br_km=(float(ib[0]), float(ib[1])))
    except Exception:
        pass

ORBIT_DATA = load_json(DATA_DIR / "orbit_schedule.json")
SENS_DATA = load_json(DATA_DIR / "sensitivity_schedule.json")

def orbit_available() -> bool:
    return isinstance(ORBIT_DATA, list) and len(ORBIT_DATA) >= 2

def sens_available() -> bool:
    return isinstance(SENS_DATA, list) and len(SENS_DATA) >= 1

# -----------------------------
# ベクトル便利
# -----------------------------
def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))

def unit(v: np.ndarray) -> np.ndarray:
    n = norm(v)
    if n < 1e-12:
        return v * 0.0
    return v / n

def angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    c = float(np.clip(np.dot(unit(u), unit(v)), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))

def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x*x*(3-2*x)

def _vec(entry: Dict[str, object], key: str) -> Optional[np.ndarray]:
    v = entry.get(key, None)
    if not isinstance(v, list):
        return None
    if len(v) == 2:
        return np.array([float(v[0]), float(v[1]), 0.0], dtype=float)
    if len(v) == 3:
        return np.array([float(v[0]), float(v[1]), float(v[2])], dtype=float)
    return None

def get_positions_3d(day: float, total_days: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if orbit_available():
        data = [d for d in ORBIT_DATA if isinstance(d, dict) and "day" in d]
        data.sort(key=lambda d: float(d.get("day", 0.0)))
        if len(data) >= 2:
            # bracket
            if day <= float(data[0]["day"]):
                e0, e1 = data[0], data[1]
            elif day >= float(data[-1]["day"]):
                e0, e1 = data[-2], data[-1]
            else:
                e0, e1 = data[0], data[1]
                for i in range(len(data)-1):
                    d0 = float(data[i]["day"]); d1 = float(data[i+1]["day"])
                    if d0 <= day <= d1:
                        e0, e1 = data[i], data[i+1]
                        break
            d0 = float(e0["day"]); d1 = float(e1["day"])
            t = 0.0 if abs(d1-d0) < 1e-12 else (day-d0)/(d1-d0)

            def lerp(key: str) -> Optional[np.ndarray]:
                v0 = _vec(e0, key); v1 = _vec(e1, key)
                if v0 is None or v1 is None:
                    return None
                return (1-t)*v0 + t*v1

            sun = lerp("sun"); earth = lerp("earth"); venus = lerp("venus"); sc = lerp("ikaros")
            if sun is not None and earth is not None and venus is not None and sc is not None:
                return sun, earth, venus, sc

    # fallback toy
    EARTH_AU = 1.0
    VENUS_AU = 0.723
    EARTH_PERIOD_D = 365.25
    VENUS_PERIOD_D = 224.7

    def planet(day_: float, a: float, P: float, phase: float) -> np.ndarray:
        th = 2*math.pi*(day_/P) + phase
        return np.array([a*math.cos(th), a*math.sin(th), 0.0], dtype=float)

    def sc_pos(day_: float) -> np.ndarray:
        f = smoothstep(day_/total_days)
        r = EARTH_AU - (EARTH_AU - VENUS_AU)*f
        omega = 2*math.pi/320.0
        th = 2*math.pi*(day_/EARTH_PERIOD_D) + omega*day_
        return np.array([r*math.cos(th), r*math.sin(th), 0.0], dtype=float)

    sun = np.array([0.0,0.0,0.0])
    earth = planet(day, EARTH_AU, EARTH_PERIOD_D, 0.0)
    venus = planet(day, VENUS_AU, VENUS_PERIOD_D, 0.6)
    sc = sc_pos(day)
    return sun, earth, venus, sc

def get_sensitivity(turn: int) -> np.ndarray:
    if sens_available():
        for d in SENS_DATA:
            if isinstance(d, dict) and int(d.get("turn", -1)) == int(turn):
                C = d.get("C", None)
                try:
                    M = np.array(C, dtype=float)
                    if M.shape == (2,2):
                        return M
                except Exception:
                    pass
    gain = 8.0 + 0.9*turn
    return gain*np.array([[1.0,0.3],[-0.2,0.9]], dtype=float)

# -----------------------------
# 幾何（通信・でんき）
# -----------------------------
def in_blackout(day: float) -> bool:
    return CFG.blackout_start_day <= day <= CFG.blackout_end_day

def beta_to_tilt(beta_in: float, beta_out: float) -> float:
    return float(math.sqrt(beta_in**2 + beta_out**2))

def power_percent(tilt_deg: float) -> float:
    return float(max(0.0, math.cos(math.radians(tilt_deg))) * 100.0)

def comm_ok(earth_aspect_deg: float) -> bool:
    return (earth_aspect_deg <= CFG.comm_ok_low_deg) or (earth_aspect_deg >= CFG.comm_ok_high_deg)

def make_local_frame(sc: np.ndarray, sun: np.ndarray, earth: np.ndarray) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    z = unit(sun - sc)        # SC→Sun
    e = unit(earth - sc)      # SC→Earth
    x_raw = e - np.dot(e, z)*z
    if norm(x_raw) < 1e-6:
        tmp = np.array([1.0,0.0,0.0])
        if abs(np.dot(tmp,z)) > 0.9:
            tmp = np.array([0.0,1.0,0.0])
        x_raw = tmp - np.dot(tmp,z)*z
    x = unit(x_raw)
    y = unit(np.cross(z, x))
    return x, y, z

def sail_normal(sc: np.ndarray, sun: np.ndarray, earth: np.ndarray, beta_in: float, beta_out: float) -> np.ndarray:
    x,y,z = make_local_frame(sc, sun, earth)
    tilt = beta_to_tilt(beta_in, beta_out)
    n_local = np.array([math.sin(math.radians(beta_in)),
                        math.sin(math.radians(beta_out)),
                        math.cos(math.radians(tilt))], dtype=float)
    n_local = unit(n_local)
    return unit(n_local[0]*x + n_local[1]*y + n_local[2]*z)

def earth_aspect(sc: np.ndarray, sun: np.ndarray, earth: np.ndarray, beta_in: float, beta_out: float) -> float:
    x,y,z = make_local_frame(sc, sun, earth)
    n = sail_normal(sc, sun, earth, beta_in, beta_out)
    e = unit(earth - sc)
    nL = np.array([np.dot(n,x), np.dot(n,y), np.dot(n,z)])
    eL = np.array([np.dot(e,x), np.dot(e,y), np.dot(e,z)])
    return angle_deg(nL, eL)

def alpha_deg(sc: np.ndarray, sun: np.ndarray, earth: np.ndarray) -> float:
    return angle_deg(sun-sc, earth-sc)

# -----------------------------
# 予測楕円（モンテカルロ）
# -----------------------------
def predict_next(x_hat: np.ndarray, C: np.ndarray, u: np.ndarray, k_hat: float, q: float, n: int) -> Tuple[np.ndarray,np.ndarray]:
    rng = np.random.default_rng(12345 + int(10*abs(u[0])+7*abs(u[1])))
    k = np.clip(rng.normal(k_hat, 0.15, size=n), 0.2, 2.0)
    w = rng.normal(0.0, q, size=(n,2))
    du = (k[:,None]*(C@u)[None,:])
    samples = x_hat[None,:] + du + w
    return samples.mean(axis=0), np.cov(samples.T)

def ellipse(mu: np.ndarray, cov: np.ndarray, nsig: float, N: int=200) -> np.ndarray:
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-9)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:,order]
    t = np.linspace(0,2*np.pi,N)
    circle = np.vstack([np.cos(t), np.sin(t)])
    pts = (vecs @ (nsig*np.sqrt(vals)[:,None]*circle)).T + mu[None,:]
    return pts

# -----------------------------
# セッション状態
# -----------------------------
def init_state(seed: int=2):
    rng = np.random.default_rng(seed)
    st.session_state.turn = 0
    st.session_state.day = 0.0
    st.session_state.rng_seed = int(seed)

    st.session_state.x_true = np.array([CFG.init_bt_br_km[0], CFG.init_bt_br_km[1]], dtype=float)
    st.session_state.x_hat = st.session_state.x_true + rng.normal(0.0, CFG.init_est_noise_km, size=2)

    st.session_state.k_true = float(rng.uniform(0.75, 1.25))
    st.session_state.k_hat = 1.0
    st.session_state.log: List[Dict[str, object]] = []

if "turn" not in st.session_state:
    init_state()

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="IKAROS-GO", layout="wide")
st.title("IKAROS-GO（試作）— IKAROSみたいに“ねらって”うごかすゲーム")

total_days = CFG.n_turns*CFG.turn_days
sun, earth, venus, sc = get_positions_3d(st.session_state.day, total_days)
alpha_now = alpha_deg(sc, sun, earth)

with st.sidebar:
    st.header("そうさ")
    st.caption("データが無ければ模型で動きます。データがあれば本物寄りに動きます。")
    st.write(f"- mission_config.json: {'ある ✅' if isinstance(MISSION, dict) else 'ない'}")
    st.write(f"- orbit_schedule.json: {'ある ✅' if orbit_available() else 'ない'}")
    st.write(f"- sensitivity_schedule.json: {'ある ✅' if sens_available() else 'ない'}")

    st.markdown("---")
    seed = st.number_input("ランダムのタネ（seed）", 0, 9999, int(st.session_state.rng_seed), 1)
    if st.button("さいしょからやり直す"):
        init_state(int(seed))
        st.rerun()

    st.markdown("---")
    beta_in = st.slider("β_in（左右）[deg]", -CFG.beta_max_deg, CFG.beta_max_deg, 0.0, CFG.beta_step_deg)
    beta_out = st.slider("β_out（上下）[deg]", -CFG.beta_max_deg, CFG.beta_max_deg, 0.0, CFG.beta_step_deg)
    n_samples = st.slider("よそうの点の数", 200, 3000, 900, 100)
    vec_scale = st.slider("3Dベクトルの長さ（見やすさ）", 0.02, 0.40, 0.12, 0.01)

    st.markdown("---")
    step_btn = st.button("つぎの2週間へ（すすめる）", type="primary", disabled=(st.session_state.turn >= CFG.n_turns))

# 状態表示
progress = min(1.0, st.session_state.turn/CFG.n_turns)
st.caption("進みぐあい（0% → 100%）")
st.progress(progress)

tilt = beta_to_tilt(beta_in, beta_out)
pwr = power_percent(tilt)
ea = earth_aspect(sc, sun, earth, beta_in, beta_out)
comm_success = (not in_blackout(st.session_state.day)) and (tilt <= CFG.sun_tilt_limit_deg) and comm_ok(ea)

c1,c2,c3,c4 = st.columns(4)
c1.metric("いまのターン", f"{st.session_state.turn+1}/{CFG.n_turns}")
c2.metric("地球と太陽の角度 α", f"{alpha_now:.1f}°")
c3.metric("発電（模型）", f"{pwr:.0f}%")
c4.metric("通信", "できた ✅" if comm_success else "できない ❌")

if in_blackout(st.session_state.day):
    st.warning("いまはブラックアウト中：なにをしても通信できません（演出）")

# すすめる
if step_btn:
    rng = np.random.default_rng(int(st.session_state.rng_seed) + 1000 + int(st.session_state.turn))
    u = np.array([beta_in, beta_out], dtype=float)

    C = get_sensitivity(int(st.session_state.turn))
    C_true = float(st.session_state.k_true) * C

    w = rng.normal(0.0, CFG.process_noise_km, size=2)
    st.session_state.x_true = st.session_state.x_true + C_true @ u + w

    if comm_success:
        meas = st.session_state.x_true + rng.normal(0.0, CFG.meas_noise_km, size=2)
        st.session_state.x_hat = 0.6*st.session_state.x_hat + 0.4*meas
        st.session_state.k_hat += 0.15*(float(st.session_state.k_true) - float(st.session_state.k_hat))

    st.session_state.log.append({
        "turn": int(st.session_state.turn),
        "day": float(st.session_state.day),
        "beta_in": float(beta_in),
        "beta_out": float(beta_out),
        "BT_hat": float(st.session_state.x_hat[0]),
        "BR_hat": float(st.session_state.x_hat[1]),
        "alpha_deg": float(alpha_now),
        "tilt_deg": float(tilt),
        "power_%": float(pwr),
        "earth_aspect_deg": float(ea),
        "comm": bool(comm_success),
        "blackout": bool(in_blackout(st.session_state.day)),
        "k_hat": float(st.session_state.k_hat),
    })

    st.session_state.turn += 1
    st.session_state.day += float(CFG.turn_days)
    if st.session_state.turn >= CFG.n_turns:
        st.success("ミッションおわり！（試作）")
    st.rerun()

# -----------------------------
# タブ
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["B-plane（ねらい）","太陽系の図（2D）","βマップ","3次元可視化"])

with tab1:
    st.subheader("B-plane（ねらいの平面）")
    x_hat = np.array(st.session_state.x_hat, dtype=float)
    u = np.array([beta_in, beta_out], dtype=float)
    C = get_sensitivity(int(st.session_state.turn)) if st.session_state.turn < CFG.n_turns else np.eye(2)

    mu, cov = predict_next(x_hat, C, u, float(st.session_state.k_hat), CFG.process_noise_km, int(n_samples))
    e1 = ellipse(mu, cov, 1.0)
    e2 = ellipse(mu, cov, 2.0)

    target = np.array(CFG.target_bt_br_km, dtype=float)
    tol = float(CFG.tolerance_km)

    fig = go.Figure()
    th = np.linspace(0,2*np.pi,240)
    fig.add_trace(go.Scatter(x=target[0]+tol*np.cos(th), y=target[1]+tol*np.sin(th), mode="lines", name="ゴール（ゆるい範囲）"))

    if st.session_state.log:
        xs = [d["BT_hat"] for d in st.session_state.log]
        ys = [d["BR_hat"] for d in st.session_state.log]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name="これまで"))

    fig.add_trace(go.Scatter(x=[x_hat[0]], y=[x_hat[1]], mode="markers", name="いま（よそう）"))
    fig.add_trace(go.Scatter(x=[float(mu[0])], y=[float(mu[1])], mode="markers", name="つぎ（まんなか）"))
    fig.add_trace(go.Scatter(x=e2[:,0], y=e2[:,1], mode="lines", name="つぎのよそう（2σ）", fill="toself", opacity=0.15))
    fig.add_trace(go.Scatter(x=e1[:,0], y=e1[:,1], mode="lines", name="つぎのよそう（1σ）", fill="toself", opacity=0.25))

    fig.update_layout(xaxis_title="B_T (km)", yaxis_title="B_R (km)", height=600, margin=dict(l=10,r=10,t=10,b=10), showlegend=True)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("ログ")
    st.dataframe(st.session_state.log, use_container_width=True)

with tab2:
    st.subheader("太陽・地球・金星・IKAROS（2D図）")
    days = np.linspace(0, total_days, 260)
    sc_p, e_p, v_p = [], [], []
    for d in days:
        s,e,v,k = get_positions_3d(float(d), total_days)
        sc_p.append(k[:2]); e_p.append(e[:2]); v_p.append(v[:2])
    sc_p = np.vstack(sc_p); e_p = np.vstack(e_p); v_p = np.vstack(v_p)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=e_p[:,0], y=e_p[:,1], mode="lines", name="地球の道"))
    fig2.add_trace(go.Scatter(x=v_p[:,0], y=v_p[:,1], mode="lines", name="金星の道"))
    fig2.add_trace(go.Scatter(x=sc_p[:,0], y=sc_p[:,1], mode="lines", name="IKAROSの道"))
    fig2.add_trace(go.Scatter(x=[0], y=[0], mode="markers", name="太陽"))
    fig2.add_trace(go.Scatter(x=[earth[0]], y=[earth[1]], mode="markers", name="地球"))
    fig2.add_trace(go.Scatter(x=[venus[0]], y=[venus[1]], mode="markers", name="金星"))
    fig2.add_trace(go.Scatter(x=[sc[0]], y=[sc[1]], mode="markers", name="IKAROS（いま）"))
    fig2.update_layout(xaxis_title="x", yaxis_title="y", height=640, margin=dict(l=10,r=10,t=10,b=10), showlegend=True)
    fig2.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig2, use_container_width=True)

    st.write(f"- きょうは **{st.session_state.day:.0f}日目**\n- 進みぐあいは **{progress*100:.0f}%**\n- 地球と太陽の角度 α は **{alpha_now:.1f}°**")

with tab3:
    st.subheader("βマップ（通信とでんき）")
    grid_step = 1.0
    xs = np.arange(-CFG.beta_max_deg, CFG.beta_max_deg + 1e-9, grid_step)
    ys = np.arange(-CFG.beta_max_deg, CFG.beta_max_deg + 1e-9, grid_step)
    P = np.zeros((len(ys), len(xs)), dtype=float)
    COMM = np.zeros((len(ys), len(xs)), dtype=float)

    if in_blackout(st.session_state.day):
        for j, bo in enumerate(ys):
            for i, bi in enumerate(xs):
                P[j,i] = power_percent(beta_to_tilt(float(bi), float(bo)))
                COMM[j,i] = 0.0
    else:
        for j, bo in enumerate(ys):
            for i, bi in enumerate(xs):
                tilt_ = beta_to_tilt(float(bi), float(bo))
                P[j,i] = power_percent(tilt_)
                ok = (tilt_ <= CFG.sun_tilt_limit_deg) and comm_ok(earth_aspect(sc, sun, earth, float(bi), float(bo)))
                COMM[j,i] = 1.0 if ok else 0.0

    fig3 = go.Figure()
    fig3.add_trace(go.Heatmap(x=xs, y=ys, z=P, colorbar=dict(title="でんき(%)")))
    fig3.add_trace(go.Contour(x=xs, y=ys, z=COMM, showscale=False, contours=dict(start=0.5,end=0.5,size=1), name="通信OKの線", line=dict(width=3)))
    th = np.linspace(0,2*np.pi,240)
    r = CFG.sun_tilt_limit_deg
    fig3.add_trace(go.Scatter(x=r*np.cos(th), y=r*np.sin(th), mode="lines", name=f"でんき制限 {r:.0f}°"))
    fig3.add_trace(go.Scatter(x=[beta_in], y=[beta_out], mode="markers", name="いまのβ"))
    fig3.update_layout(xaxis_title="β_in (deg)", yaxis_title="β_out (deg)", height=650, margin=dict(l=10,r=10,t=10,b=10), showlegend=True)
    fig3.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig3, use_container_width=True)

    st.write(f"- 太陽からのかたむき: **{tilt:.1f}°**\n- でんき: **{pwr:.0f}%**\n- 地球に向けた角度: **{ea:.1f}°**\n- 通信: **{'できる' if comm_success else 'できない'}**")

with tab4:
    st.subheader("3次元可視化（ベクトルつき）")
    days = np.linspace(0, total_days, 260)
    sc_p, e_p, v_p = [], [], []
    for d in days:
        s,e,v,k = get_positions_3d(float(d), total_days)
        sc_p.append(k); e_p.append(e); v_p.append(v)
    sc_p = np.vstack(sc_p); e_p = np.vstack(e_p); v_p = np.vstack(v_p)

    sun_dir = unit(sun - sc)
    earth_dir = unit(earth - sc)
    sail_dir = sail_normal(sc, sun, earth, beta_in, beta_out)
    L = float(vec_scale)

    def vec_trace(name: str, v: np.ndarray):
        p0 = sc
        p1 = sc + L*v
        return go.Scatter3d(x=[p0[0],p1[0]], y=[p0[1],p1[1]], z=[p0[2],p1[2]], mode="lines+markers", name=name)

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter3d(x=e_p[:,0], y=e_p[:,1], z=e_p[:,2], mode="lines", name="地球の道"))
    fig4.add_trace(go.Scatter3d(x=v_p[:,0], y=v_p[:,1], z=v_p[:,2], mode="lines", name="金星の道"))
    fig4.add_trace(go.Scatter3d(x=sc_p[:,0], y=sc_p[:,1], z=sc_p[:,2], mode="lines", name="IKAROSの道"))
    fig4.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers", name="太陽"))
    fig4.add_trace(go.Scatter3d(x=[earth[0]], y=[earth[1]], z=[earth[2]], mode="markers", name="地球"))
    fig4.add_trace(go.Scatter3d(x=[venus[0]], y=[venus[1]], z=[venus[2]], mode="markers", name="金星"))
    fig4.add_trace(go.Scatter3d(x=[sc[0]], y=[sc[1]], z=[sc[2]], mode="markers", name="IKAROS（いま）"))

    fig4.add_trace(vec_trace("太陽方向（SC→Sun）", sun_dir))
    fig4.add_trace(vec_trace("地球方向（SC→Earth）", earth_dir))
    fig4.add_trace(vec_trace("帆面法線（β）", sail_dir))

    fig4.update_layout(scene=dict(xaxis_title="x",yaxis_title="y",zaxis_title="z", aspectmode="data"),
                       height=720, margin=dict(l=10,r=10,t=10,b=10), showlegend=True)
    st.plotly_chart(fig4, use_container_width=True)

    st.write(
        "- **太陽方向**：こっちへ向けると、でんきが作りやすい\n"
        "- **地球方向**：こっちへ向けると、通信しやすい\n"
        "- **帆面法線**：βで決まる“いまのむき”（操作してるやつ）"
    )

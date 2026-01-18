\
"""
IKAROSっぽい（地球→金星）に近い “それっぽい” データを作るツール。

生成物（--out に出る）
- orbit_schedule.json : sun/earth/venus/ikaros の 3D位置（AU）
- sensitivity_schedule.json : turnごとの 2×2 感度行列（km/deg）
- mission_config.json : アプリ設定（ターン数や初期B-planeなど）

例:
  python tools/generate_data.py --out data --profile ikaros2010 --step 2
"""

from __future__ import annotations

import argparse, json, math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np

AU_KM = 149_597_870.7
K_GAUSS = 0.01720209895
MU_SUN = K_GAUSS**2  # AU^3/day^2


def jd_from_utc(dt: datetime) -> float:
    year, month = dt.year, dt.month
    day = dt.day + (dt.hour + (dt.minute + dt.second/60)/60)/24
    if month <= 2:
        year -= 1
        month += 12
    A = year // 100
    B = 2 - A + (A // 4)
    jd = int(365.25*(year+4716)) + int(30.6001*(month+1)) + day + B - 1524.5
    return float(jd)


@dataclass
class PlanetElements:
    a: float
    e: float
    i_deg: float
    L_deg: float
    varpi_deg: float
    Omega_deg: float


EARTH = PlanetElements(a=1.00000011, e=0.01671022, i_deg=0.00005,
                       L_deg=100.46435, varpi_deg=102.94719, Omega_deg=-11.26064)
VENUS = PlanetElements(a=0.72333199, e=0.00677323, i_deg=3.39471,
                       L_deg=181.97973, varpi_deg=131.53298, Omega_deg=76.68069)


def kepler_E(M: float, e: float) -> float:
    E = M
    for _ in range(30):
        f = E - e*math.sin(E) - M
        fp = 1 - e*math.cos(E)
        E -= f/(fp+1e-12)
    return E

def rot3(a: float) -> np.ndarray:
    c,s = math.cos(a), math.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=float)

def rot1(a: float) -> np.ndarray:
    c,s = math.cos(a), math.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=float)

def planet_state(el: PlanetElements, jd: float, jd0: float=2451545.0) -> Tuple[np.ndarray,np.ndarray]:
    a,e = el.a, el.e
    i = math.radians(el.i_deg)
    L0 = math.radians(el.L_deg)
    varpi = math.radians(el.varpi_deg)
    Omega = math.radians(el.Omega_deg)
    omega = varpi - Omega
    M0 = L0 - varpi

    n = math.sqrt(MU_SUN/(a**3))
    t = jd - jd0
    M = (M0 + n*t) % (2*math.pi)
    E = kepler_E(M,e)
    nu = 2*math.atan2(math.sqrt(1+e)*math.sin(E/2), math.sqrt(1-e)*math.cos(E/2))
    r = a*(1 - e*math.cos(E))
    r_pf = np.array([r*math.cos(nu), r*math.sin(nu), 0.0], dtype=float)

    p = a*(1-e**2)
    h = math.sqrt(MU_SUN*p)
    v_pf = np.array([-MU_SUN/h*math.sin(nu), MU_SUN/h*(e+math.cos(nu)), 0.0], dtype=float)

    Q = rot3(Omega) @ rot1(i) @ rot3(omega)
    return Q@r_pf, Q@v_pf


def stumpff_C(z: float) -> float:
    if z > 1e-8:
        s = math.sqrt(z)
        return (1-math.cos(s))/z
    if z < -1e-8:
        s = math.sqrt(-z)
        return (math.cosh(s)-1)/(-z)
    return 0.5

def stumpff_S(z: float) -> float:
    if z > 1e-8:
        s = math.sqrt(z)
        return (s-math.sin(s))/(s**3)
    if z < -1e-8:
        s = math.sqrt(-z)
        return (math.sinh(s)-s)/(s**3)
    return 1/6


def lambert_universal(r1: np.ndarray, r2: np.ndarray, dt: float, mu: float=MU_SUN, prograde: bool=True) -> Tuple[np.ndarray,np.ndarray]:
    """
    Universal-variable Lambert (single-rev) with safe bracketing + bisection.
    r1,r2: AU, dt: days, returns v1,v2 in AU/day
    """
    r1n = float(np.linalg.norm(r1)); r2n = float(np.linalg.norm(r2))
    cosd = float(np.dot(r1,r2)/(r1n*r2n+1e-12))
    cosd = max(-1.0, min(1.0, cosd))
    dth = math.acos(cosd)
    cr = np.cross(r1,r2)
    if prograde and cr[2] < 0:
        dth = 2*math.pi - dth
    if (not prograde) and cr[2] >= 0:
        dth = 2*math.pi - dth

    A = math.sin(dth)*math.sqrt(r1n*r2n/(1-math.cos(dth)+1e-12))
    if abs(A) < 1e-10:
        raise RuntimeError("Lambert failed (A too small)")

    def time_of_flight(z: float) -> Tuple[float, float]:
        # return (t, y). if invalid, t=inf
        try:
            C = stumpff_C(z); S = stumpff_S(z)
            y = r1n + r2n + A*(z*S - 1)/math.sqrt(C+1e-12)
            if y < 0:
                return float("inf"), y
            x = math.sqrt(y/(C+1e-12))
            t = (x**3*S + A*math.sqrt(y))/math.sqrt(mu)
            if not math.isfinite(t):
                return float("inf"), y
            return t, y
        except OverflowError:
            return float("inf"), float("nan")

    # --- bracket by scanning z range ---
    def find_bracket(zmin: float, zmax: float, N: int=600) -> Tuple[float,float]:
        zs = np.linspace(zmin, zmax, N)
        prev = None
        prev_z = None
        for z in zs:
            t,_ = time_of_flight(float(z))
            if not math.isfinite(t) or t == float("inf"):
                continue
            val = t - dt
            if prev is None:
                prev, prev_z = val, float(z)
                continue
            if prev == 0:
                return prev_z, prev_z
            if val == 0:
                return float(z), float(z)
            if (prev < 0 and val > 0) or (prev > 0 and val < 0):
                return prev_z, float(z)
            prev, prev_z = val, float(z)
        raise RuntimeError("Lambert bracket not found (try different dates/range)")

    # Try moderate then wider range
    try:
        z0, z1 = find_bracket(-30.0, 30.0, 800)
    except Exception:
        z0, z1 = find_bracket(-120.0, 120.0, 1200)

    # --- bisection ---
    for _ in range(80):
        zm = 0.5*(z0+z1)
        tm,_ = time_of_flight(zm)
        if not math.isfinite(tm) or tm == float("inf"):
            # push toward smaller magnitude
            z1 = zm
            continue
        fm = tm - dt
        if abs(fm) < 1e-8:
            z0 = z1 = zm
            break
        t0,_ = time_of_flight(z0)
        f0 = t0 - dt
        if (f0 < 0 and fm > 0) or (f0 > 0 and fm < 0):
            z1 = zm
        else:
            z0 = zm

    z = 0.5*(z0+z1)
    t, y = time_of_flight(z)
    C = stumpff_C(z); S = stumpff_S(z)
    f = 1 - y/r1n
    g = A*math.sqrt(y/mu)
    gdot = 1 - y/r2n
    v1 = (r2 - f*r1)/(g+1e-12)
    v2 = (gdot*r2 - r1)/(g+1e-12)
    return v1, v2



def kepler_propagate(r0: np.ndarray, v0: np.ndarray, dt: float, mu: float=MU_SUN) -> Tuple[np.ndarray,np.ndarray]:
    r0n = float(np.linalg.norm(r0))
    v0n = float(np.linalg.norm(v0))
    vr0 = float(np.dot(r0,v0)/(r0n+1e-12))
    alpha = 2/r0n - (v0n**2)/mu

    chi = math.sqrt(mu)*abs(alpha)*dt if abs(alpha) > 1e-8 else math.sqrt(mu)*dt/r0n

    for _ in range(80):
        z = alpha*chi**2
        C = stumpff_C(z); S = stumpff_S(z)
        dt_est = (chi**3*S + (vr0/math.sqrt(mu))*chi**2*C + r0n*chi*(1 - z*S))/math.sqrt(mu)
        f = dt_est - dt
        if abs(f) < 1e-10:
            break
        dtdchi = (chi**2*C + (vr0/math.sqrt(mu))*chi*(1 - z*S) + r0n*(1 - z*C))/math.sqrt(mu)
        chi -= f/(dtdchi+1e-12)

    z = alpha*chi**2
    C = stumpff_C(z); S = stumpff_S(z)
    f = 1 - (chi**2/r0n)*C
    g = dt - (chi**3/math.sqrt(mu))*S
    r = f*r0 + g*v0
    rn = float(np.linalg.norm(r))
    fdot = (math.sqrt(mu)/(rn*r0n))*(z*S - 1)*chi
    gdot = 1 - (chi**2/rn)*C
    v = fdot*r0 + gdot*v0
    return r, v


def srp_accel(r: np.ndarray, beta_in: float, beta_out: float, a0: float) -> np.ndarray:
    # sun at origin, SC->Sun = -r
    z = (-r)/(np.linalg.norm(r)+1e-12)
    tmp = np.array([1.0,0.0,0.0])
    if abs(np.dot(tmp,z)) > 0.9:
        tmp = np.array([0.0,1.0,0.0])
    x = tmp - np.dot(tmp,z)*z
    x = x/(np.linalg.norm(x)+1e-12)
    y = np.cross(z,x)

    tilt = math.sqrt(beta_in**2 + beta_out**2)
    n_local = np.array([math.sin(math.radians(beta_in)),
                        math.sin(math.radians(beta_out)),
                        math.cos(math.radians(tilt))], dtype=float)
    n = (n_local[0]*x + n_local[1]*y + n_local[2]*z)
    n = n/(np.linalg.norm(n)+1e-12)
    return a0*math.cos(math.radians(tilt))*n


def integrate_srp(r0: np.ndarray, v0: np.ndarray, t0: float, t1: float, h: float, a0: float, beta_in: float, beta_out: float) -> Tuple[np.ndarray,np.ndarray]:
    r = r0.copy(); v = v0.copy(); t = t0

    def acc(rr: np.ndarray, vv: np.ndarray) -> np.ndarray:
        rn = np.linalg.norm(rr)+1e-12
        a_g = -MU_SUN*rr/(rn**3)
        a_s = srp_accel(rr, beta_in, beta_out, a0)
        return a_g + a_s

    while t < t1 - 1e-12:
        dt = min(h, t1-t)
        # RK4
        k1r = v
        k1v = acc(r,v)
        k2r = v + 0.5*dt*k1v
        k2v = acc(r + 0.5*dt*k1r, v + 0.5*dt*k1v)
        k3r = v + 0.5*dt*k2v
        k3v = acc(r + 0.5*dt*k2r, v + 0.5*dt*k2v)
        k4r = v + dt*k3v
        k4v = acc(r + dt*k3r, v + dt*k3v)

        r = r + (dt/6)*(k1r + 2*k2r + 2*k3r + k4r)
        v = v + (dt/6)*(k1v + 2*k2v + 2*k3v + k4v)
        t += dt
    return r, v


def bplane_like(r_sc: np.ndarray, v_sc: np.ndarray, r_v: np.ndarray, v_v: np.ndarray) -> Tuple[float,float]:
    r_rel = r_sc - r_v
    v_rel = v_sc - v_v
    S = v_rel/(np.linalg.norm(v_rel)+1e-12)
    k = np.array([0.0,0.0,1.0])
    T = np.cross(k,S)
    if np.linalg.norm(T) < 1e-8:
        k = np.array([0.0,1.0,0.0]); T = np.cross(k,S)
    T = T/(np.linalg.norm(T)+1e-12)
    R = np.cross(S,T)
    BT = float(np.dot(r_rel,T))*AU_KM
    BR = float(np.dot(r_rel,R))*AU_KM
    return BT, BR


def generate_ikaros2010(out_dir: Path, step_days: float):
    # 実ミッション日付（公開情報）に合わせる
    launch = datetime(2010,5,21,0,0,0,tzinfo=timezone.utc)
    flyby  = datetime(2010,12,8,0,0,0,tzinfo=timezone.utc)

    jd0 = jd_from_utc(launch)
    jd1 = jd_from_utc(flyby)
    tof = jd1 - jd0  # days

    n_turns = 14
    turn_days = 14

    rE0, vE0 = planet_state(EARTH, jd0)
    rV1, vV1 = planet_state(VENUS, jd1)

    v1, v2 = lambert_universal(rE0, rV1, tof, MU_SUN, prograde=True)

    # orbit schedule (AU)
    times = np.arange(0.0, tof + 1e-9, step_days)
    rows: List[dict] = []
    for t in times:
        jd = jd0 + t
        rE,_ = planet_state(EARTH, jd)
        rV,_ = planet_state(VENUS, jd)
        rS,_ = kepler_propagate(rE0, v1, t, MU_SUN)
        rows.append({
            "day": float(t),
            "sun": [0.0,0.0,0.0],
            "earth": [float(rE[0]), float(rE[1]), float(rE[2])],
            "venus": [float(rV[0]), float(rV[1]), float(rV[2])],
            "ikaros": [float(rS[0]), float(rS[1]), float(rS[2])],
        })

    # baseline at encounter
    rV_end, vV_end = planet_state(VENUS, jd1)
    rS_end, vS_end = kepler_propagate(rE0, v1, tof, MU_SUN)
    BT0, BR0 = bplane_like(rS_end, vS_end, rV_end, vV_end)

    # sensitivity: SRP only during each turn
    a0 = 2.0e-7  # AU/day^2 (toy)
    h = 0.5       # day
    sens: List[dict] = []

    for k in range(n_turns):
        t0 = k*turn_days
        t1 = min((k+1)*turn_days, tof)
        r0, v0 = kepler_propagate(rE0, v1, t0, MU_SUN)

        def run(bi: float, bo: float) -> Tuple[float,float]:
            r_seg, v_seg = integrate_srp(r0, v0, t0, t1, h, a0, bi, bo)
            r_end, v_end = kepler_propagate(r_seg, v_seg, tof - t1, MU_SUN)
            return bplane_like(r_end, v_end, rV_end, vV_end)

        d = 1.0
        BT1, BR1 = run(d,0.0)
        BT2, BR2 = run(0.0,d)
        C = [[(BT1-BT0)/d, (BT2-BT0)/d],
             [(BR1-BR0)/d, (BR2-BR0)/d]]
        sens.append({"turn": int(k), "C": [[float(C[0][0]), float(C[0][1])],[float(C[1][0]), float(C[1][1])]]})

    mission = {
        "profile": "ikaros2010",
        "launch_utc": launch.isoformat(),
        "flyby_utc": flyby.isoformat(),
        "n_turns": n_turns,
        "turn_days": turn_days,
        "tolerance_km": 30.0,
        "sun_tilt_limit_deg": 45.0,
        "target_bt_br_km": [0.0,0.0],
        "init_bt_br_km": [float(BT0), float(BR0)],
        "blackout_start_day": 100.0,
        "blackout_end_day": 130.0,
        "units": {"orbit":"AU","bplane":"km","beta":"deg"},
        "notes": "Toy: planet is simple Kepler; transfer is Lambert; sensitivity uses SRP during each turn only."
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/"orbit_schedule.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out_dir/"sensitivity_schedule.json").write_text(json.dumps(sens, indent=2), encoding="utf-8")
    (out_dir/"mission_config.json").write_text(json.dumps(mission, indent=2), encoding="utf-8")

    print("Generated in", out_dir)
    print("Initial B-plane-like (km):", BT0, BR0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data")
    ap.add_argument("--profile", type=str, default="ikaros2010")
    ap.add_argument("--step", type=float, default=2.0)
    args = ap.parse_args()

    out = Path(args.out)
    if args.profile == "ikaros2010":
        generate_ikaros2010(out, float(args.step))
    else:
        raise SystemExit("Unknown profile: " + args.profile)

if __name__ == "__main__":
    main()

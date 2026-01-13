\
# Math notes (simplified)

This prototype uses a **minimal** model intended for fast iteration and easy deployment.
It does **not** perform full orbit propagation. Instead, it treats the Venus encounter
targeting point on the **B-plane** as the game state.

## State and control

- State (B-plane coordinates):
  \[
    x_k =
    \begin{bmatrix}
      B_T \\
      B_R
    \end{bmatrix}
  \]
  where \(B_T, B_R\) are the transverse/radial coordinates on the B-plane (km).

- Control (attitude command):
  \[
    u_k =
    \begin{bmatrix}
      \Delta \beta_{in} \\
      \Delta \beta_{out}
    \end{bmatrix}
  \]
  where \(\beta\) angles are treated as player inputs (deg).

## Discrete-time dynamics

The game advances in *turns* (default: 2 weeks). The core model is:

\[
x_{k+1} = x_k + C_k u_k + w_k
\]

- \(C_k\) is a 2×2 **sensitivity** matrix mapping attitude changes to B-plane shifts.
- \(w_k\) is a small process noise term (unmodeled effects).

### Where does \(C_k\) come from?

In a more realistic pipeline, you would precompute a mapping from attitude history
to encounter parameters using state transition matrices (STM) or variational equations
around a reference trajectory. In this prototype we simply:

- use a toy \(C_k\) that grows later in the mission (to mimic “some phases are more effective”),
- and optionally scale it by an unknown “sail efficiency” factor \(k_{true}\).

## Estimation (“OD”) and delayed feedback

Operationally, you don’t always have continuous contact. So the prototype models:

- **True state**: \(x^{true}_k\)
- **Estimated state**: \(x^{hat}_k\)

If communication link is available, we simulate an observation:

\[
y_k = x^{true}_k + v_k
\]

and update the estimate by a simple blending rule (not a full Kalman filter).

If comm is unavailable, **no update** happens — players must act under uncertainty.

## Constraints used in the game

These constraints are implemented as “gates” that disable control and/or OD updates:

- Power: sun angle `< 45°`
- Communication: earth aspect angle in `[0,60]` or `[120,180]` degrees
- Optional blackout window (toy event) for extra realism

## Extending toward a more realistic model

Replace the toy components with your mission-accurate precomputed data:

- Sun / Earth aspect angle tables per turn
- Sensitivity matrices \(C_k\) per turn (or per smaller segment)
- Better estimator (EKF/UKF) if you want players to feel OD uncertainty more strongly

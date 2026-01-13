\
# Customization guide

You can replace the toy schedules with your own precomputed tables, while keeping the Streamlit UI.

## 1) Replace sun/earth angles with `data/angles_schedule.json`

Create a JSON file:

```json
[
  {"turn": 0, "day": 0,  "sun_angle_deg": 35.0, "earth_angle_deg": 120.0},
  {"turn": 1, "day": 14, "sun_angle_deg": 36.2, "earth_angle_deg": 130.5}
]
```

The app will pick the closest entry by `day`.

## 2) Replace sensitivity matrices with `data/sensitivity_schedule.json`

Create a JSON file:

```json
[
  {"turn": 0, "C": [[1.0, 0.3], [-0.2, 0.9]]},
  {"turn": 1, "C": [[1.1, 0.2], [-0.1, 1.0]]}
]
```

The app will use the entry whose `turn` matches the current turn, and scale it by the hidden
efficiency factor `k_true` (you can remove scaling if you want).

## 3) What if I want STM-based mapping?

A practical pipeline for the “real” version:

1. Pick a reference Earth–Venus transfer and a nominal attitude schedule.
2. Propagate variational equations / compute STM segments.
3. Build a mapping from small changes in β-in/out to changes in B-plane coordinates.
4. Export per-turn matrices into `sensitivity_schedule.json`.

This preserves the **fast UI loop** while keeping the physics behind the scenes.

## 4) Knobs that dramatically change gameplay

- Increase OD noise -> more “ops pain”
- Add command delay (execute u_k only after next comm window)
- Limit total |Δβ| budget per mission
- Add score terms: “hit target”, “minimize RCS”, “maximize contact time”

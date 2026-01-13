\
# IKAROS-GO (prototype)

A lightweight, Streamlit-deployable parody/learning game inspired by **HTV-GO!**:
you “pilot” a solar-sail spacecraft (IKAROS-like) by adjusting attitude angles
(β-in / β-out) to steer the **Venus B-plane** targeting point, while experiencing
operational constraints like **communications windows** and **power limits**.

This repository is intentionally **simple** (no full orbit propagation) so you can
deploy quickly and iterate on the gameplay logic.

> Want to make it more “real”? Replace the toy schedules with your precomputed
> state transition / sensitivity data (see `docs/customize.md`).

---

## Quick start (local)

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Deploy on Streamlit Community Cloud (fast path)

1. Push this repo to GitHub.
2. Go to Streamlit Community Cloud and create a new app from the repo.
3. Set:
   - **Main file path**: `app.py`
4. Deploy.

No extra config is required.

---

## What’s inside

- `app.py` — the Streamlit app (turn-based guidance game)
- `requirements.txt` — minimal Python dependencies
- `docs/math.md` — the simplified math model (B-plane state + sensitivity)
- `docs/customize.md` — how to feed real schedules (`data/*.json`)
- `docs/game_design.md` — gameplay ideas & knobs
- `data/` — placeholder examples of schedule files

---

## Model (prototype)

- State: `x_k = [B_T, B_R]^T` (km), a 2D point on the B-plane.
- Control: `u_k = [Δβ_in, Δβ_out]^T` (deg), attitude change commands.
- Dynamics (toy):
  ```
  x_{k+1} = x_k + C_k u_k + w_k
  ```
- Constraints (simplified):
  - Power: sun angle must be `< 45°`
  - Comms: earth aspect angle in `[0,60]` or `[120,180]` degrees
  - Optional blackout event window (toy) to emulate “no-link” periods

See `docs/math.md` for details.

---

## References (starting points)

- ISSFD 2011 paper (IKAROS Venus B-plane targeting operations / guidance concepts):
  - https://issfd.org/ISSFD_2011/S3-Interplanetary.Mission.Design.1-IMD1/S3_P6_ISSFD22_PF_075.pdf
- J-STAGE paper (IKAROS comm/antenna operational constraints, etc.):
  - https://www.jstage.jst.go.jp/article/kjsass/61/4/61_KJ00008636303/_pdf/-char/ja

---

## License

MIT (see `LICENSE`).

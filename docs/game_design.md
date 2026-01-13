\
# Game design notes (ideas)

This prototype is intentionally minimal. Here are upgrade ideas that preserve “fun + learning”:

## Turn structure (ops feeling)

- Each turn = 2 weeks
  - First half: attitude command
  - Second half: orbit determination (if comm available)

Delayed feedback is one of the most educational parts.

## Missions & difficulty

- Easy: always-on comm, no blackout window, generous target tolerance
- Normal: comm angle gate + power gate
- Hard: add blackout windows + command delay + bigger disturbances

## Scoring

- Primary: final distance to target on B-plane
- Secondary:
  - number of comm opportunities used
  - total attitude change magnitude (proxy for consumables)
  - maximum sun angle margin (power safety)

## UI upgrades

- Show predicted impact vector (“if you command this, expected move is …”)
- Show uncertainty ellipse for x_hat
- Add a “timeline” bar of comm windows / blackout windows

## Physics upgrades (optional)

- Replace C_k with precomputed STM-based matrices
- Add coupling from attitude -> sun angle (power) and/or antenna pointing
- Add “sail efficiency estimation” mini-game (OD improves it gradually)

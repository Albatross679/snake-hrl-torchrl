---
date_created: 2026-02-18
date_modified: 2026-02-18
tags: [knowledge, physics, locomotion, pipe, gravity, friction, contact]
---

# Pipe Locomotion Without Gravity

Gravity is not required for snake locomotion inside pipes. On open ground, gravity provides the normal force that enables friction (`N = mg`). In a pipe, the snake's own body curvature presses against the walls, generating normal forces from elastic confinement — not from weight.

## Why It Works

1. The CPG/controller sets a target curvature that tries to bend the snake body
2. The pipe wall prevents the body from expanding beyond the pipe radius
3. The contact penalty force provides a reaction (normal force on the wall)
4. Friction at the contact points converts the serpentine wave into forward thrust

`N_wall = f(body curvature, pipe radius, rod stiffness)` — independent of gravity.

A straight snake in a pipe has zero wall contact and cannot move. A snake with active curvature grips the walls and propels itself. The snake controls its own grip strength by modulating wave amplitude.

## Simulation Setting

Set `enable_gravity = False` for pipe navigation. Gravity only adds asymmetry (more bottom contact in horizontal pipes) without contributing to the locomotion mechanism.

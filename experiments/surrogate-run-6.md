# Surrogate Run 6

## Previous Run Summary (Run 5: surrogate_20260317_000319)
- **Epochs:** 80 (early stopped, patience=30 triggered at epoch 50)
- **Best val_loss:** 0.7123 (epoch 50)
- **Final train_loss:** 1.037
- **Train-val gap:** 0.31 (clear overfitting — train kept dropping, val diverged)
- **Component losses (final):**
  - omega_z: 1.96 (dominant, known hard — see issues/surrogate-omega-z-poor-prediction.md)
  - vel_y: 0.241
  - vel_x: 0.092
  - yaw: 0.063
  - heading: 0.003
  - rel_pos_x/y: ~6e-5 / ~3e-5
  - com: ~3e-7
- **Config:** 4x1024 MLP, dropout=0.0, weight_decay=1e-5, lr=1e-4

## Changes Made
1. **dropout: 0.0 → 0.1** — add dropout regularization to reduce overfitting
2. **weight_decay: 1e-5 → 1e-4** — 10x stronger L2 regularization

## Hypothesis
The train-val gap (0.31) indicates the model is memorizing training data rather than generalizing. Adding dropout and stronger weight decay should:
- Delay or prevent the val_loss plateau/rise
- Allow training to continue improving val_loss past epoch 50
- Potentially achieve val_loss < 0.70
- omega_z remains the dominant component and won't be fundamentally fixed by regularization alone (needs architectural changes per the issue doc), but reducing overfitting should help all components.

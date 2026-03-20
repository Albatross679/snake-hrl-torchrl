# Issue: Checkpoint shape mismatch when resuming from 113722

## Date: 2026-03-17

## Error
```
RuntimeError: Error(s) in loading state_dict for SurrogateModel:
    size mismatch for mlp.0.weight: copying a param with shape torch.Size([1024, 137]) from checkpoint, the shape in current model is torch.Size([1024, 139]).
    size mismatch for mlp.16.weight: copying a param with shape torch.Size([128, 1024]) from checkpoint, the shape in current model is torch.Size([130, 1024]).
    size mismatch for mlp.16.bias: copying a param with shape torch.Size([128]) from checkpoint, the shape in current model is torch.Size([130]).
```

## Root Cause
- Checkpoint `output/surrogate_20260317_113722` was trained on `data/surrogate_rl_step_rel128` (state_dim=128)
- Current code uses `data/surrogate_rl_step` which produces state_dim=130 after `raw_to_relative()` conversion
- The 2 extra dims (128→130) propagate to input_dim (137→139) and output_dim (128→130)

## Resolution
- Cannot fine-tune from old checkpoint; started fresh training instead
- Future: could add `strict=False` + partial weight loading for cross-architecture transfer

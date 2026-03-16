"""Neural network surrogate for DisMech Discrete Elastic Rod dynamics.

Replaces the inner-loop physics computation of the DisMech implicit Euler
solver with a trained MLP that predicts the next rod state from
(current_state, action, serpenoid_time) in a single forward pass. Enables
GPU-batched environments for orders-of-magnitude faster RL training.

Modules:
    state           -- RodState2D pack/unpack (DisMech extraction), StateNormalizer
    collect_config  -- Data collection config
    train_config    -- Model, training, env, and RL configs
    model           -- SurrogateModel (nn.Module MLP)
    dataset         -- FlatStepDataset (torch Dataset)
    collect_data    -- Data collection from real DisMech env
    train_surrogate -- Surrogate model training
    env             -- SurrogateLocomotionEnv (TorchRL EnvBase, GPU-batched)
    validate        -- Accuracy validation vs real env
"""

from aprx_model_dismech.state import RodState2D, StateNormalizer
from aprx_model_dismech.model import SurrogateModel

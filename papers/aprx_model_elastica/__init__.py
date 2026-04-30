"""Neural network surrogate for PyElastica Cosserat rod dynamics.

Replaces the inner-loop physics computation in LocomotionElasticaEnv with a
trained MLP that predicts the next rod state from (current_state, action,
serpenoid_time) in a single forward pass. Enables GPU-batched environments
for orders-of-magnitude faster RL training.

Modules:
    state           -- RodState2D pack/unpack, StateNormalizer
    collect_config  -- Data collection config
    train_config    -- Model, training, env, and RL configs
    model           -- SurrogateModel (nn.Module MLP)
    dataset         -- SurrogateDataset (torch Dataset)
    collect_data    -- Data collection from real PyElastica env
    train_surrogate -- Surrogate model training
    env             -- SurrogateLocomotionEnv (TorchRL EnvBase, GPU-batched)
    train_rl        -- RL training with surrogate env
    validate        -- Accuracy validation vs real env
"""

from aprx_model_elastica.state import RodState2D, StateNormalizer
from aprx_model_elastica.model import SurrogateModel

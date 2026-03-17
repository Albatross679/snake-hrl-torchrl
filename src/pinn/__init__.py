"""PINN (Physics-Informed Neural Network) package for snake rod surrogate models.

Provides physics regularization, adaptive loss balancing, and nondimensionalization
utilities for training physics-informed surrogates of Cosserat rod dynamics.
"""

from src.pinn.regularizer import PhysicsRegularizer
from src.pinn.loss_balancing import ReLoBRaLo
from src.pinn.nondim import NondimScales
from src.pinn.physics_residual import CosseratRHS
from src.pinn.collocation import sample_collocation, adaptive_refinement

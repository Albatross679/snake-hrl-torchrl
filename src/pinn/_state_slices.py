"""Rod state vector layout constants (mirrors papers.aprx_model_elastica.state).

Duplicated here to avoid importing the full papers.aprx_model_elastica package
which triggers a deep import chain requiring torchrl, elastica, etc. These
values are simple constants that rarely change.
"""

NUM_NODES = 21
NUM_ELEMENTS = 20
NUM_JOINTS = 19
RAW_STATE_DIM = 124

# Named slices into the raw 124-dim state vector
POS_X = slice(0, 21)
POS_Y = slice(21, 42)
VEL_X = slice(42, 63)
VEL_Y = slice(63, 84)
YAW = slice(84, 104)
OMEGA_Z = slice(104, 124)

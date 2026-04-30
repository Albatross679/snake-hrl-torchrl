"""Generate staggered grid diagram for the discrete Cosserat rod."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

N_SHOW = 7
Ne = N_SHOW - 1
Nv = Ne - 1

node_x = np.arange(N_SHOW, dtype=float) * 1.2
elem_x = 0.5 * (node_x[:-1] + node_x[1:])
vor_x = 0.5 * (elem_x[:-1] + elem_x[1:])

y_node = 2.0
y_elem = 1.0
y_vor = 0.0

fig, ax = plt.subplots(figsize=(16, 10))

# --------------------------------------------------------------------------
# Rod centerline
# --------------------------------------------------------------------------
ax.plot(node_x, [y_node] * N_SHOW, '-', color='#dddddd', linewidth=10, zorder=0)

# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------
for i, x in enumerate(node_x):
    ax.plot(x, y_node, 'o', color='#2166ac', markersize=16, zorder=3)
    ax.text(x, y_node + 0.30, f'$n_{{{i+1}}}$', ha='center', va='bottom',
            fontsize=12, color='#2166ac', fontweight='bold')

# --------------------------------------------------------------------------
# Elements
# --------------------------------------------------------------------------
for e, x in enumerate(elem_x):
    ax.plot(x, y_elem, 's', color='#b2182b', markersize=14, zorder=3)
    ax.text(x, y_elem - 0.30, f'$e_{{{e+1}}}$', ha='center', va='top',
            fontsize=12, color='#b2182b', fontweight='bold')
    # Connecting lines to endpoint nodes
    ax.plot([node_x[e], x], [y_node - 0.18, y_elem + 0.18], '-',
            color='#bbbbbb', linewidth=1, zorder=1)
    ax.plot([node_x[e + 1], x], [y_node - 0.18, y_elem + 0.18], '-',
            color='#bbbbbb', linewidth=1, zorder=1)

# --------------------------------------------------------------------------
# Voronoi nodes
# --------------------------------------------------------------------------
for v, x in enumerate(vor_x):
    ax.plot(x, y_vor, 'D', color='#1a9850', markersize=12, zorder=3)
    ax.text(x, y_vor - 0.30, f'$v_{{{v+1}}}$', ha='center', va='top',
            fontsize=12, color='#1a9850', fontweight='bold')
    ax.plot([elem_x[v], x], [y_elem - 0.18, y_vor + 0.18], '--',
            color='#bbbbbb', linewidth=1, zorder=1)
    ax.plot([elem_x[v + 1], x], [y_elem - 0.18, y_vor + 0.18], '--',
            color='#bbbbbb', linewidth=1, zorder=1)

# --------------------------------------------------------------------------
# Ellipsis
# --------------------------------------------------------------------------
for y in [y_node, y_elem, y_vor]:
    ax.text(node_x[-1] + 0.35, y, r'$\cdots$', fontsize=18, ha='left',
            va='center', color='#888888')

# --------------------------------------------------------------------------
# Row labels (left side)
# --------------------------------------------------------------------------
ax.text(-1.0, y_node, 'Nodes\n($N = 21$)', ha='right', va='center',
        fontsize=12, fontweight='bold', color='#2166ac')
ax.text(-1.0, y_elem, 'Elements\n($N_e = 20$)', ha='right', va='center',
        fontsize=12, fontweight='bold', color='#b2182b')
ax.text(-1.0, y_vor, 'Voronoi\n($N_v = 19$)', ha='right', va='center',
        fontsize=12, fontweight='bold', color='#1a9850')

# --------------------------------------------------------------------------
# Operator arrows (left margin)
# --------------------------------------------------------------------------
arr_x = -0.4
mid_ne = 0.5 * (y_node + y_elem)
ax.annotate('', xy=(arr_x - 0.08, y_elem + 0.30),
            xytext=(arr_x - 0.08, y_node - 0.30),
            arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5))
ax.text(arr_x - 0.25, mid_ne, r'$\mathbf{D}$', ha='right', va='center',
        fontsize=13, color='#555555')

ax.annotate('', xy=(arr_x + 0.08, y_node - 0.30),
            xytext=(arr_x + 0.08, y_elem + 0.30),
            arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5))
ax.text(arr_x + 0.25, mid_ne, r'$\mathbf{D}^T$', ha='left', va='center',
        fontsize=13, color='#555555')

mid_ev = 0.5 * (y_elem + y_vor)
ax.annotate('', xy=(arr_x - 0.08, y_vor + 0.30),
            xytext=(arr_x - 0.08, y_elem - 0.30),
            arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5))
ax.text(arr_x - 0.25, mid_ev, r'$\mathbf{A}_v$', ha='right', va='center',
        fontsize=13, color='#555555')

ax.annotate('', xy=(arr_x + 0.08, y_elem - 0.30),
            xytext=(arr_x + 0.08, y_vor + 0.30),
            arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5))
ax.text(arr_x + 0.25, mid_ev, r'$\mathbf{D}_v$', ha='left', va='center',
        fontsize=13, color='#555555')

# --------------------------------------------------------------------------
# Legend boxes BELOW the diagram
# --------------------------------------------------------------------------
box_y_start = -1.2
box_w = 0.30  # fraction of fig width per box
lx = [0.02, 0.35, 0.68]  # left x positions (axes fraction)

node_text = (
    r'$\mathbf{Nodes}$ carry:' '\n'
    r'  $x_i, y_i$ — position' '\n'
    r'  $\dot{x}_i, \dot{y}_i$ — velocity' '\n'
    r'  $m_i$ — lumped mass' '\n'
    r'  (84 of 124 state vars)'
)

elem_text = (
    r'$\mathbf{Elements}$ carry:' '\n'
    r'  $\psi_e$ — yaw angle' '\n'
    r'  $\omega_{z,e}$ — angular velocity' '\n'
    r'  $\ell_e, \varepsilon_e, \mathbf{t}_e$ — geometry (derived)' '\n'
    r'  $\boldsymbol{\sigma}_e, \mathbf{n}_e^L$ — strain/stress (derived)' '\n'
    r'  (40 of 124 state vars)'
)

vor_text = (
    r'$\mathbf{Voronoi}$ $\mathbf{nodes}$ carry:' '\n'
    r'  $\kappa_v$ — curvature (derived)' '\n'
    r'  $\kappa_v^{\mathrm{rest}}$ — rest curvature (CPG input)' '\n'
    r'  $\boldsymbol{\tau}_v^L$ — bending couple (derived)' '\n'
    r'  $\hat{\varepsilon}_v, \hat{\ell}_v^{\mathrm{rest}}$ — Voronoi geometry' '\n'
    r'  (0 independent state vars)'
)

texts = [
    (node_text, '#d1e5f0', '#2166ac'),
    (elem_text, '#fddbc7', '#b2182b'),
    (vor_text,  '#d9f0d3', '#1a9850'),
]

for idx, (txt, bg, tc) in enumerate(texts):
    bx = lx[idx] * (node_x[-1] + 1.5) - 0.5
    ax.text(bx, box_y_start, txt, fontsize=10, va='top', ha='left',
            color='#333333',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=bg, alpha=0.85,
                      edgecolor=tc, linewidth=1.5))

# --------------------------------------------------------------------------
# Title
# --------------------------------------------------------------------------
ax.set_title('Staggered Grid Discretization of the Cosserat Rod\n'
             '(showing 7 of 21 nodes)',
             fontsize=15, fontweight='bold', pad=15)

# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------
ax.set_xlim(-1.8, node_x[-1] + 1.5)
ax.set_ylim(box_y_start - 2.8, y_node + 0.9)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
plt.savefig('/home/user/snake-hrl-torchrl/figures/staggered_grid.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig('/home/user/snake-hrl-torchrl/figures/staggered_grid.pdf',
            bbox_inches='tight', facecolor='white')
print("Saved staggered_grid.png and staggered_grid.pdf")

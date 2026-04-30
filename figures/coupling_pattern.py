"""Generate coupling pattern diagram showing how adjacent nodes interact."""

import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                         gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.35})

spacing = 1.8
nodes_x = np.arange(5, dtype=float) * spacing
elem_x = 0.5 * (nodes_x[:-1] + nodes_x[1:])
vor_x = 0.5 * (elem_x[:-1] + elem_x[1:])

MS_N = 22   # node marker size
MS_E = 18   # element marker size
MS_V = 15   # voronoi marker size
FS_LBL = 11  # label font inside marker
FS_ANN = 10  # annotation font

# =========================================================================
# Panel A: Shear-Stretch Coupling
# =========================================================================
ax = axes[0]
ax.set_title(
    r'(a)  Shear–Stretch Coupling  (nearest neighbor, via $\mathbf{D}$ and $\mathbf{D}^T$)',
    fontsize=13, fontweight='bold', loc='left', pad=12)

y_n = 1.2
y_e = -0.4

# Nodes
for i, x in enumerate(nodes_x):
    is_source = (i == 2)
    is_affected = (i in [1, 3])
    ec = '#d6604d' if (is_source or is_affected) else 'none'
    ew = 3.0 if is_affected else (0 if not is_source else 0)
    fc = '#d6604d' if is_source else '#2166ac'
    ax.plot(x, y_n, 'o', color=fc, markersize=MS_N, zorder=5,
            markeredgecolor=ec if is_affected else 'none',
            markeredgewidth=ew)
    ax.text(x, y_n, f'$n_{{{i+1}}}$', ha='center', va='center',
            fontsize=FS_LBL, color='white', fontweight='bold', zorder=6)

# Elements
for e, x in enumerate(elem_x):
    highlight = (e in [1, 2])
    fc = '#e34a33' if highlight else '#b2182b'
    ax.plot(x, y_e, 's', color=fc, markersize=MS_E, zorder=5)
    ax.text(x, y_e, f'$e_{{{e+1}}}$', ha='center', va='center',
            fontsize=FS_LBL - 1, color='white', fontweight='bold', zorder=6)

# D arrows: n3 -> e2, e3 (strain depends on position difference)
for e_idx in [1, 2]:
    x_e = elem_x[e_idx]
    x_left = nodes_x[e_idx]
    x_right = nodes_x[e_idx + 1]
    ax.annotate('', xy=(x_e - 0.15, y_e + 0.30),
                xytext=(x_left + 0.20, y_n - 0.30),
                arrowprops=dict(arrowstyle='->', color='#ef8a62', lw=2.0,
                                connectionstyle='arc3,rad=0.1'))
    ax.annotate('', xy=(x_e + 0.15, y_e + 0.30),
                xytext=(x_right - 0.20, y_n - 0.30),
                arrowprops=dict(arrowstyle='->', color='#ef8a62', lw=2.0,
                                connectionstyle='arc3,rad=-0.1'))

# D^T arrows: e2, e3 -> n2, n3, n4 (forces distributed back)
for e_idx, target_nodes in [(1, [1, 2]), (2, [2, 3])]:
    x_e = elem_x[e_idx]
    for ni in target_nodes:
        x_n = nodes_x[ni]
        ax.annotate('', xy=(x_n + (0.15 if x_n < x_e else -0.15), y_n - 0.30),
                    xytext=(x_e + (-0.15 if x_n < x_e else 0.15), y_e + 0.30),
                    arrowprops=dict(arrowstyle='->', color='#67a9cf', lw=1.8,
                                    linestyle='--',
                                    connectionstyle='arc3,rad=0.08'))

# Annotation box (right side, vertically centered)
ax.text(nodes_x[-1] + 1.0, 0.5 * (y_n + y_e),
        r'Moving $n_3$ changes strain in $e_2, e_3$' '\n'
        r'$\rightarrow$ forces redistribute to $n_2, n_3, n_4$' '\n\n'
        r'Coupling range: $\pm 1$ node' '\n'
        r'Jacobian: tridiagonal',
        fontsize=FS_ANN, va='center', ha='left',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#fee0d2',
                  edgecolor='#ef8a62', alpha=0.9, linewidth=1.5))

# Legend
from matplotlib.lines import Line2D
leg_elements = [
    Line2D([0], [0], color='#ef8a62', lw=2, label=r'$\mathbf{D}$: position diff $\to$ strain'),
    Line2D([0], [0], color='#67a9cf', lw=2, linestyle='--',
           label=r'$\mathbf{D}^T$: stress $\to$ nodal forces'),
]
ax.legend(handles=leg_elements, loc='upper left', fontsize=9.5, framealpha=0.9)

ax.set_xlim(-1.0, nodes_x[-1] + 5.5)
ax.set_ylim(-1.3, 2.2)
ax.set_aspect('equal')
ax.axis('off')

# =========================================================================
# Panel B: Bending-Twist Coupling
# =========================================================================
ax = axes[1]
ax.set_title(
    r'(b)  Bending–Twist Coupling  (next-nearest neighbor, via $\mathbf{A}_v$ and $\mathbf{D}_v$)',
    fontsize=13, fontweight='bold', loc='left', pad=12)

y_n = 2.5
y_e = 1.3
y_v = 0.0

# Nodes
for i, x in enumerate(nodes_x):
    is_source = (i == 2)
    is_affected = (i in [0, 1, 3, 4])
    fc = '#d6604d' if is_source else '#2166ac'
    ec = '#d6604d' if is_affected else 'none'
    ew = 3.0 if is_affected else 0
    ax.plot(x, y_n, 'o', color=fc, markersize=MS_N, zorder=5,
            markeredgecolor=ec, markeredgewidth=ew)
    ax.text(x, y_n, f'$n_{{{i+1}}}$', ha='center', va='center',
            fontsize=FS_LBL, color='white', fontweight='bold', zorder=6)

# Elements
for e, x in enumerate(elem_x):
    ax.plot(x, y_e, 's', color='#b2182b', markersize=MS_E, zorder=5)
    ax.text(x, y_e, f'$e_{{{e+1}}}$', ha='center', va='center',
            fontsize=FS_LBL - 1, color='white', fontweight='bold', zorder=6)

# Voronoi
for v, x in enumerate(vor_x):
    ax.plot(x, y_v, 'D', color='#1a9850', markersize=MS_V, zorder=5)
    ax.text(x, y_v, f'$v_{{{v+1}}}$', ha='center', va='center',
            fontsize=FS_LBL - 1, color='white', fontweight='bold', zorder=6)

# Node -> Element connections (all, light)
for e in range(len(elem_x)):
    ax.plot([nodes_x[e], elem_x[e]], [y_n - 0.25, y_e + 0.25], '-',
            color='#cccccc', lw=1, zorder=1)
    ax.plot([nodes_x[e + 1], elem_x[e]], [y_n - 0.25, y_e + 0.25], '-',
            color='#cccccc', lw=1, zorder=1)

# Element -> Voronoi connections (all, light)
for v in range(len(vor_x)):
    ax.plot([elem_x[v], vor_x[v]], [y_e - 0.25, y_v + 0.25], '--',
            color='#cccccc', lw=1, zorder=1)
    ax.plot([elem_x[v + 1], vor_x[v]], [y_e - 0.25, y_v + 0.25], '--',
            color='#cccccc', lw=1, zorder=1)

# Highlight the active chain: n3 -> e2,e3 -> v1,v2,v3
for e_idx in [1, 2]:
    ax.plot([nodes_x[2], elem_x[e_idx]], [y_n - 0.25, y_e + 0.25], '-',
            color='#ef8a62', lw=2.5, zorder=2)
# e1,e2 -> v1; e2,e3 -> v2; e3,e4 -> v3
for v_idx in range(3):
    for e_off in [0, 1]:
        e_src = v_idx + e_off
        if e_src < len(elem_x):
            ax.plot([elem_x[e_src], vor_x[v_idx]], [y_e - 0.25, y_v + 0.25],
                    '--', color='#66c2a5', lw=2.0, zorder=2)

# Annotation box
ax.text(nodes_x[-1] + 1.0, y_e,
        r'Moving $n_3$ changes $\psi$ in $e_2, e_3$' '\n'
        r'$\rightarrow$ curvature $\kappa$ at $v_1, v_2, v_3$' '\n'
        r'$\rightarrow$ torques on $e_1$ through $e_4$' '\n'
        r'$\rightarrow$ affects $n_1$ through $n_5$' '\n\n'
        r'Coupling range: $\pm 2$ nodes' '\n'
        r'Jacobian: pentadiagonal',
        fontsize=FS_ANN, va='center', ha='left',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#d9f0d3',
                  edgecolor='#1a9850', alpha=0.9, linewidth=1.5))

# Layer labels
ax.text(-0.8, y_n, 'Nodes', ha='right', va='center', fontsize=10,
        fontweight='bold', color='#2166ac')
ax.text(-0.8, y_e, 'Elements', ha='right', va='center', fontsize=10,
        fontweight='bold', color='#b2182b')
ax.text(-0.8, y_v, 'Voronoi', ha='right', va='center', fontsize=10,
        fontweight='bold', color='#1a9850')

ax.set_xlim(-1.5, nodes_x[-1] + 5.5)
ax.set_ylim(-0.8, 3.3)
ax.set_aspect('equal')
ax.axis('off')

plt.savefig('/home/user/snake-hrl-torchrl/figures/coupling_pattern.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig('/home/user/snake-hrl-torchrl/figures/coupling_pattern.pdf',
            bbox_inches='tight', facecolor='white')
print("Saved coupling_pattern.png and coupling_pattern.pdf")

"""Generate surrogate training result figures for the report."""
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Load metrics
metrics_path = "output/surrogate_20260317_113722/metrics.jsonl"
data = []
with open(metrics_path) as f:
    for line in f:
        data.append(json.loads(line))

epochs = [d['epoch'] for d in data]
train_loss = [d['train_loss'] for d in data]
val_loss = [d['val_loss'] for d in data]
r2_overall = [d['r2'] for d in data]

# Per-component R²
r2_com = [d['r2/com'] for d in data]
r2_heading = [d['r2/heading'] for d in data]
r2_rel_pos = [d['r2/rel_pos'] for d in data]
r2_vel = [d['r2/vel'] for d in data]
r2_yaw = [d['r2/yaw'] for d in data]
r2_omega = [d['r2/omega_z'] for d in data]

# --- Figure 1: Training and validation loss curves ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

ax1.plot(epochs, train_loss, label='Train loss', color='#2563eb', linewidth=1.2)
ax1.plot(epochs, val_loss, label='Validation loss', color='#dc2626', linewidth=1.2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('MSE Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, max(epochs))

ax2.plot(epochs, r2_overall, label='Overall $R^2$', color='#16a34a', linewidth=1.5)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('$R^2$')
ax2.set_title('Overall $R^2$ Score')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, max(epochs))
ax2.set_ylim(0, 0.5)

plt.tight_layout()
plt.savefig('figures/surrogate_training_curves.pdf')
plt.savefig('figures/surrogate_training_curves.png')
print("Saved: figures/surrogate_training_curves.pdf")

# --- Figure 2: Per-component R² bar chart (final epoch) ---
fig, ax = plt.subplots(figsize=(6, 3.5))

components = ['CoM\nposition', 'Relative\npositions', 'Angular\nvelocity $\\omega_z$', 'Heading\n$\\theta$', 'Yaw\n$\\psi_e$', 'Velocities\n$\\dot{x}, \\dot{y}$']
r2_values = [r2_com[-1], r2_rel_pos[-1], r2_omega[-1], r2_heading[-1], r2_yaw[-1], r2_vel[-1]]

# Sort by R² descending
order = np.argsort(r2_values)[::-1]
components = [components[i] for i in order]
r2_values = [r2_values[i] for i in order]

colors = ['#16a34a' if v > 0.5 else '#eab308' if v > 0.2 else '#dc2626' for v in r2_values]

bars = ax.barh(range(len(components)), r2_values, color=colors, edgecolor='white', height=0.6)
ax.set_yticks(range(len(components)))
ax.set_yticklabels(components)
ax.set_xlabel('$R^2$ Score')
ax.set_title('Per-Component Prediction Accuracy (Epoch 110)')
ax.set_xlim(0, 1.0)
ax.invert_yaxis()
ax.grid(True, axis='x', alpha=0.3)

# Add value labels
for bar, val in zip(bars, r2_values):
    ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
            va='center', ha='left', fontsize=9)

plt.tight_layout()
plt.savefig('figures/surrogate_component_r2.pdf')
plt.savefig('figures/surrogate_component_r2.png')
print("Saved: figures/surrogate_component_r2.pdf")

# --- Figure 3: Per-component R² evolution over epochs ---
fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(epochs, r2_com, label='CoM position', linewidth=1.3, color='#16a34a')
ax.plot(epochs, r2_rel_pos, label='Relative positions', linewidth=1.3, color='#2563eb')
ax.plot(epochs, r2_omega, label='$\\omega_z$', linewidth=1.3, color='#9333ea')
ax.plot(epochs, r2_heading, label='Heading $\\theta$', linewidth=1.3, color='#ea580c')
ax.plot(epochs, r2_yaw, label='Yaw $\\psi_e$', linewidth=1.3, color='#0891b2')
ax.plot(epochs, r2_vel, label='Velocities', linewidth=1.3, color='#dc2626')
ax.plot(epochs, r2_overall, label='Overall', linewidth=2, color='black', linestyle='--')

ax.set_xlabel('Epoch')
ax.set_ylabel('$R^2$ Score')
ax.set_title('Per-Component $R^2$ Evolution During Training')
ax.legend(loc='center right', fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, max(epochs))
ax.set_ylim(-0.05, 1.0)

plt.tight_layout()
plt.savefig('figures/surrogate_r2_evolution.pdf')
plt.savefig('figures/surrogate_r2_evolution.png')
print("Saved: figures/surrogate_r2_evolution.pdf")

# Print summary table for LaTeX
print("\n=== FINAL EPOCH SUMMARY (Epoch 110) ===")
print(f"Overall R²: {r2_overall[-1]:.4f}")
print(f"Overall val_loss: {val_loss[-1]:.4f}")
print(f"\nPer-component R² (sorted):")
names = ['CoM position', 'Relative positions', 'Angular velocity ω_z', 'Heading θ', 'Yaw ψ_e', 'Velocities']
vals = [r2_com[-1], r2_rel_pos[-1], r2_omega[-1], r2_heading[-1], r2_yaw[-1], r2_vel[-1]]
for n, v in sorted(zip(names, vals), key=lambda x: -x[1]):
    print(f"  {n:25s}: {v:.4f}")

# Per-component MSE at final epoch
print(f"\nPer-component MSE:")
final = data[-1]
for key in ['com', 'heading', 'rel_pos_x', 'rel_pos_y', 'vel_x', 'vel_y', 'yaw', 'omega_z']:
    print(f"  component/{key:12s}: {final[f'component/{key}']:.6f}")

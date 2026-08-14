"""
Poster figure (Option 1 only): Stacked bar chart comparing task-by-task learning
from scratch vs factorised transfer.

Generates a single-panel figure showing cumulative training time per task.
"""

import matplotlib.pyplot as plt
import numpy as np

# Set style for clean, poster-friendly figures
plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.2,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
})

# Number of tasks
n_tasks = 5
tasks = np.arange(1, n_tasks + 1)

# Simulate cumulative training time for "from scratch" baseline
time_per_task_scratch = 10
cumulative_scratch = time_per_task_scratch * tasks

# Simulate cumulative training time for factorised transfer method
time_first_task = 10
time_subsequent_tasks = [3, 2.5, 2, 1.5]
cumulative_transfer = [time_first_task]
for t in time_subsequent_tasks:
    cumulative_transfer.append(cumulative_transfer[-1] + t)
cumulative_transfer = np.array(cumulative_transfer)

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(12, 5))

# For each task, show "from scratch" vs "factorised"
x_positions = np.arange(n_tasks)
bar_width = 0.35

# From scratch: full bar (all learned from scratch)
bars_scratch = ax.bar(x_positions - bar_width/2, cumulative_scratch, 
                      bar_width, 
                      color='#D55E00', 
                      edgecolor='black', 
                      linewidth=1.2,
                      label='Typical learning from scratch')

# Factorised: stacked bar showing reused vs new learning
reused_portion = cumulative_transfer - np.array([time_first_task] + time_subsequent_tasks)
new_learning_portion = np.array([time_first_task] + time_subsequent_tasks)

bars_reused = ax.bar(x_positions + bar_width/2, reused_portion, 
                     bar_width, 
                     color='#0072B2', 
                     edgecolor='black', 
                     linewidth=1.2,
                     label='Across-task knowledge reuse')

bars_new = ax.bar(x_positions + bar_width/2, new_learning_portion, 
                  bar_width, 
                  bottom=reused_portion,
                  color='#56B4E9', 
                  edgecolor='black', 
                  linewidth=1.2,
                  label='New task-specific learning')

ax.set_ylabel('Cumulative training time', fontsize=13)
ax.set_title('The idea of Transfer Learning to Speed up Learning in an Environment', fontsize=14, fontweight='bold')
ax.set_xticks(x_positions)
ax.set_xticklabels([f'Task {i}' for i in tasks])
ax.legend(loc='upper left', frameon=True, edgecolor='black', framealpha=0.8)
ax.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.8)
ax.set_ylim(0, max(cumulative_scratch) * 1.15)

# Add annotation for redundant learning
ax.annotate('Redundant\nre-learning across tasks', 
            xy=(tasks[3] - bar_width/2, cumulative_scratch[3] * 0.6),
            xytext=(tasks[3] - bar_width/2 - 1.5, cumulative_scratch[3] * 0.8),
            fontsize=10,
            ha='center',
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.7))

plt.tight_layout()

# Save as high-resolution PNG and PDF
output_png = 'option1_stacked_bars.png'

fig.savefig(output_png, dpi=300, bbox_inches='tight')

print(f"Saved: {output_png}")
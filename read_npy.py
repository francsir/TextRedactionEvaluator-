import numpy as np
import matplotlib.pyplot as plt

file_path = "pos_distance.npy"

loaded_data = np.load(file_path, allow_pickle=True).item()

num_pos_tags = len(loaded_data)
num_cols = (num_pos_tags+1) // 2
num_rows = 2

fig, axes = plt.subplots(num_rows,num_cols, figsize=(15,7))
axes = axes.flatten()

for i, (pos_tag, values) in enumerate(loaded_data.items()):
    ax = axes[i]
    ax.hist(values, bins=20, density=False, alpha=0.7, label=pos_tag)
    ax.set_title(pos_tag)
    ax.legend()

# Adjust layout
plt.tight_layout()

# Show the plot
plt.savefig("pos_distance_distros_False.png", dpi=100)

fig, axes = plt.subplots(num_rows,num_cols, figsize=(15,7))
axes = axes.flatten()

for i, (pos_tag, values) in enumerate(loaded_data.items()):
    ax = axes[i]
    ax.hist(values, bins=20, density=True, alpha=0.7, label=pos_tag)
    ax.set_title(pos_tag)
    ax.legend()

# Adjust layout
plt.tight_layout()

# Show the plot
plt.savefig("pos_distance_distros_True.png", dpi=100)
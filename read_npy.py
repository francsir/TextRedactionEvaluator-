import numpy as np
import matplotlib.pyplot as plt
import os
FOLDER = 'Data'

def __main__():
    for f in os.listdir(FOLDER):
        
        paths = [f"{FOLDER}/{f}/pos_distance.npy", f"{FOLDER}/{f}/mean_squared_distances.npy"]

        for file_path in paths:
            loaded_data = np.load(file_path, allow_pickle=True).item()

            try:
                num_pos_tags = len(loaded_data)
                num_cols = (num_pos_tags+1) // 2
                num_rows = 2

                fig, axes = plt.subplots(num_rows,num_cols, figsize=(15,7))
                axes = axes.flatten()
            except:
                continue

            for i, (pos_tag, values) in enumerate(loaded_data.items()):
                ax = axes[i]

                if np.all(np.isnan(values)):
                    continue
                

                ax.hist(values, bins=20, density=False, alpha=0.7, label=pos_tag)
                ax.set_title(pos_tag)
                ax.legend()

            # Adjust layout
            plt.tight_layout()

            # Show the plot
            plt.savefig(f"{file_path}_False.png", dpi=100)

            fig, axes = plt.subplots(num_rows,num_cols, figsize=(15,7))
            axes = axes.flatten()

            for i, (pos_tag, values) in enumerate(loaded_data.items()):
                ax = axes[i]

                if np.all(np.isnan(values)):
                    continue
                
                ax.hist(values, bins=20, density=True, alpha=0.7, label=pos_tag)
                ax.set_title(pos_tag)
                ax.legend()

            # Adjust layout
            plt.tight_layout()

            # Show the plot
            plt.savefig(f"{file_path}_True.png", dpi=100)
            plt.close('all')



import matplotlib.pyplot as plt
import torch
import numpy as np
import os


def plot_scene(obs, pred, adj=None, sample_idx=0, title="Social Scene", save_path=None):
    """
    Plots a single scene from a batch.
    Args:
        obs: List of tensors, we pick obs[sample_idx] -> [Num_Peds, T_obs, 2]
        pred: List of tensors, we pick pred[sample_idx] -> [Num_Peds, T_pred, 2]
        adj: List of tensors, we pick adj[sample_idx] -> [T_obs, Num_Peds, Num_Peds] (optional)
        sample_idx: Which index in the batch list to plot.
    """
    obs_scene = obs[sample_idx].numpy()
    pred_scene = pred[sample_idx].numpy()
    
    num_peds = obs_scene.shape[0]
    
    plt.figure(figsize=(10, 10))
    
    # 1. Plot Trajectories
    for i in range(num_peds):
        # Observed (solid blue)
        plt.plot(obs_scene[i, :, 0], obs_scene[i, :, 1], 'b-', linewidth=2, label='Observed' if i==0 else "")
        # Future (dashed green)
        plt.plot(pred_scene[i, :, 0], pred_scene[i, :, 1], 'g--', linewidth=2, label='Ground Truth' if i==0 else "")
        # Mark the current position (end of observation)
        plt.plot(obs_scene[i, -1, 0], obs_scene[i, -1, 1], 'bo', markersize=8)

    # 2. Plot Social Graph (edges at the last observed frame)
    if adj is not None:
        # Get adjacency matrix for the last observed timestep (t_obs - 1)
        last_obs_adj = adj[sample_idx][-1].numpy()
        current_pos = obs_scene[:, -1, :] # [Num_Peds, 2]
        
        for i in range(num_peds):
            for j in range(i + 1, num_peds):
                # If an edge exists (and it's not a self-loop)
                if last_obs_adj[i, j] > 0:
                    # Draw a faint red line between them
                    plt.plot([current_pos[i, 0], current_pos[j, 0]],
                             [current_pos[i, 1], current_pos[j, 1]],
                             'r-', alpha=0.3, linewidth=1)

    plt.title(title)
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal') # Crucial for spatial data so 1m x = 1m y
    # plt.show()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"📈 Plot saved to: {save_path}")
        plt.close() # Close to free up memory
    else:
        plt.show()

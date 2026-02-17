import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from src.data_pipeline.dataset import social_collate


def visualize_model_results(model, dataset, num_samples=3, save_dir="results/plots"):
    """
    Takes a trained model and plots Ground Truth vs Best-of-K predictions.
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    # We use a DataLoader with batch_size 1 to grab individual social scenes
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=social_collate)
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
                
            # 1. Prepare Inputs
            obs_norm = [o.to(next(model.parameters()).device) for o in batch['obs']]
            obs_rel = [r.to(next(model.parameters()).device) for r in batch['obs_rel']]
            target_abs = batch['pred'][0].numpy() # [N, T, 2]
            origin = batch['origin'][0].numpy()   # [N, 1, 2]
            obs_abs = (batch['obs'][0] + batch['origin'][0]).numpy()
            
            # 2. Forward Pass (K=20 for benchmark standard)
            # Returns list of [K, N, T, 2]
            preds_rel_list = model(obs_norm, obs_rel, k=20)
            preds_rel = preds_rel_list[0].cpu().numpy()
            
            # 3. Reconstruct Absolute Coordinates for all K samples
            # Cumulative sum of deltas + last observed position
            preds_abs = np.cumsum(preds_rel, axis=2) + origin # [K, N, T, 2]
            
            # 4. Find the "Best" sample (the one reported in ADE/FDE)
            # Calculate ADE for each of the K samples to pick the winner for viz
            ade_per_sample = np.mean(np.linalg.norm(preds_abs - target_abs, axis=-1), axis=(1, 2))
            best_idx = np.argmin(ade_per_sample)
            best_pred = preds_abs[best_idx]
            
            # 5. Plotting
            plt.figure(figsize=(12, 6))
            
            # Left Plot: Ground Truth
            plt.subplot(1, 2, 1)
            for p in range(obs_abs.shape[0]):
                plt.plot(obs_abs[p, :, 0], obs_abs[p, :, 1], 'b-', alpha=0.6)
                plt.plot(target_abs[p, :, 0], target_abs[p, :, 1], 'g--')
                plt.scatter(obs_abs[p, -1, 0], obs_abs[p, -1, 1], c='blue', s=30)
            plt.title(f"Scene {i}: Ground Truth")
            plt.axis('equal')
            plt.grid(True, alpha=0.3)

            # Right Plot: Prediction (Best of 20)
            plt.subplot(1, 2, 2)
            for p in range(obs_abs.shape[0]):
                # Plot the observed path
                plt.plot(obs_abs[p, :, 0], obs_abs[p, :, 1], 'b-', alpha=0.3)
                # Plot the best predicted path in Orange
                plt.plot(best_pred[p, :, 0], best_pred[p, :, 1], 'r-', linewidth=2)
                # Mark the end point
                plt.scatter(best_pred[p, -1, 0], best_pred[p, -1, 1], c='red', s=30)
                
                # Faintly plot a few other samples to show multimodality
                for k_idx in range(min(5, 20)):
                    plt.plot(preds_abs[k_idx, p, :, 0], preds_abs[k_idx, p, :, 1], 'r-', alpha=0.05)
            
            plt.title(f"Scene {i}: Best-of-20 Prediction")
            plt.axis('equal')
            plt.grid(True, alpha=0.3)
            
            save_path = f"{save_dir}/scene_{i}_eval.png"
            plt.savefig(save_path)
            plt.close()
            print(f"✅ Saved visualization to {save_path}")

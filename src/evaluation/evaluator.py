import torch
from src.evaluation.metrics import calculate_best_of_k


class Evaluator:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Standard benchmark uses K=20
        self.k = config.get('evaluation', {}).get('k_samples', 20)

    def evaluate(self, data_loader):
        self.model.eval()
        total_ade = 0
        total_fde = 0
        total_batches = 0

        with torch.no_grad():
            for batch in data_loader:
                # Unpack the richer batch from our new dataset
                obs_norm = [o.to(self.device) for o in batch['obs']]
                target_abs = [t.to(self.device) for t in batch['pred']]
                obs_rel = [r.to(self.device) for r in batch['obs_rel']]
                origins = [orig.to(self.device) for orig in batch['origin']]
                
                # Forward pass: We expect the model to handle 'K' samples
                # If your model isn't stochastic yet, it will just return [1, N, T, 2]
                preds_rel_list = self.model(obs_norm, obs_rel, k=self.k)
                
                batch_ade = 0
                batch_fde = 0
                
                for i in range(len(preds_rel_list)):
                    # 1. Reconstruct Absolute Coordinates
                    # pred_rel is [K, N, T, 2], origin is [N, 1, 2]
                    # We use torch.cumsum to turn offsets back into positions
                    preds_abs = torch.cumsum(preds_rel_list[i], dim=2) + origins[i]
                    
                    # 2. Best-of-K calculation
                    batch_ade += calculate_best_of_k(preds_abs, target_abs[i], metric_type='ade')
                    batch_fde += calculate_best_of_k(preds_abs, target_abs[i], metric_type='fde')

                total_ade += (batch_ade / len(preds_rel_list)).item()
                total_fde += (batch_fde / len(preds_rel_list)).item()
                total_batches += 1

        return {
            "ADE": total_ade / total_batches, 
            "FDE": total_fde / total_batches
        }

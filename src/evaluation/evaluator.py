import torch
from src.evaluation.metrics import calculate_ade, calculate_fde


class Evaluator:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def evaluate(self, data_loader):
        self.model.eval()
        total_ade = 0
        total_fde = 0
        total_samples = 0

        with torch.no_grad():
            for batch in data_loader:
                obs_list, target_list, adj_list = batch
                
                # Move tensors to device (if needed)
                obs_list = [o.to(self.device) for o in obs_list]
                target_list = [t.to(self.device) for t in target_list]
                
                # Forward pass
                pred_list = self.model(obs_list, adj_list)
                
                # Calculate metrics for each scene in the batch
                for pred, target in zip(pred_list, target_list):
                    total_ade += calculate_ade(pred, target).item()
                    total_fde += calculate_fde(pred, target).item()
                    total_samples += 1

        avg_ade = total_ade / total_samples
        avg_fde = total_fde / total_samples
        
        return {"ADE": avg_ade, "FDE": avg_fde}

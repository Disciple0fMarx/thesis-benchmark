import torch


def displacement_error(pred, target):
    """
    Standard Euclidean distance.
    Args:
        pred: [N, T, 2]
        target: [N, T, 2]
    """
    return torch.norm(pred - target, p=2, dim=-1)


def calculate_ade(pred, target):
    """
    Average Displacement Error for a single prediction.
    Args:
        pred/target: [N, T, 2] or list of [N, T, 2]
    """
    if isinstance(pred, list):
        # Flatten list of tensors into one large 'N' dimension
        pred = torch.cat(pred, dim=0)
        target = torch.cat(target, dim=0)
        
    errors = displacement_error(pred, target) # [N, T]
    return torch.mean(errors) # Mean across all peds and all time steps


def calculate_fde(pred, target):
    """Final Displacement Error"""
    if isinstance(pred, list):
        pred = torch.cat(pred, dim=0)
        target = torch.cat(target, dim=0)
        
    # Grab only the last timestep: [N, 1, 2]
    errors = displacement_error(pred[:, -1:, :], target[:, -1:, :])
    return torch.mean(errors)


def calculate_best_of_k(preds, target, metric_type='ade'):
    """
    Standard ETH/UCY Benchmark Metric: Best-of-K.
    Args:
        preds: [K, N, T, 2] - K different predicted futures
        target: [N, T, 2]    - The single ground truth future
        metric_type: 'ade' or 'fde'
    Returns:
        The error of the sample that is closest to the target.
    """
    k = preds.shape[0]
    errors = []
    
    for i in range(k):
        if metric_type == 'ade':
            # Calculate mean error per pedestrian for this sample
            # [N, T] -> [N]
            err = torch.mean(displacement_error(preds[i], target), dim=1)
        else:
            # [N, 1] -> [N]
            err = displacement_error(preds[i][:, -1:, :], target[:, -1:, :]).squeeze()
        errors.append(err)
    
    # errors shape: [K, N]
    errors = torch.stack(errors)
    
    # For each pedestrian, find the minimum error among the K samples
    min_errors, _ = torch.min(errors, dim=0) # [N]
    
    # Return the average of those minimums
    return torch.mean(min_errors)

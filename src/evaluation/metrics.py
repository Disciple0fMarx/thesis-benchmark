import torch

def displacement_error(pred, target):
    """
    Calculates the Euclidean distance between pred and target.
    Args:
        pred: [N, T, 2] 
        target: [N, T, 2]
    Returns:
        [N, T] distance matrix
    """
    return torch.norm(pred - target, p=2, dim=-1)


def calculate_ade(pred, target):
    """Average Displacement Error"""
    # Mean over time, then mean over pedestrians
    errors = displacement_error(pred, target)
    return torch.mean(errors)


def calculate_fde(pred, target):
    """Final Displacement Error"""
    # Distance at the last timestep
    errors = displacement_error(pred[:, -1:, :], target[:, -1:, :])
    return torch.mean(errors)

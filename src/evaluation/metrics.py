import torch
import numpy as np


def displacement_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Per-timestep Euclidean distance between predicted and ground-truth positions.

    Parameters
    ----------
    pred   : FloatTensor [N, T, 2]
    target : FloatTensor [N, T, 2]

    Returns
    -------
    FloatTensor [N, T]
    """
    return torch.norm(pred - target, p=2, dim=-1)


def calculate_ade(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Average Displacement Error for a single deterministic prediction.

    ADE = mean Euclidean distance over all pedestrians and all future timesteps.

    Parameters
    ----------
    pred   : FloatTensor [N, T, 2]  — predicted positions in world coordinates
    target : FloatTensor [N, T, 2]  — ground-truth positions in world coordinates

    Returns
    -------
    Scalar FloatTensor
    """
    return displacement_error(pred, target).mean()


def calculate_fde(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Final Displacement Error for a single deterministic prediction.

    FDE = mean Euclidean distance at the last predicted timestep.

    Parameters
    ----------
    pred   : FloatTensor [N, T, 2]
    target : FloatTensor [N, T, 2]

    Returns
    -------
    Scalar FloatTensor
    """
    # Index the last timestep explicitly — squeeze(-1) would be ambiguous
    # when N=1 (single-pedestrian scene), and squeeze() with no argument
    # would collapse that dimension silently.
    return displacement_error(pred[:, -1, :], target[:, -1, :]).mean()


def calculate_best_of_k(
    preds: torch.Tensor,
    target: torch.Tensor,
    metric_type: str = 'ade',
) -> torch.Tensor:
    """
    Best-of-K (minADE / minFDE) — the standard ETH-UCY benchmark metric.

    For each pedestrian, we find which of the K predicted futures is closest
    to the ground truth, then average those minimum errors across all
    pedestrians.  This is the metric reported by Social-GAN, STGCNN, LED,
    MoFlow, and every other paper in the benchmark.

    Parameters
    ----------
    preds       : FloatTensor [K, N, T, 2]  — K predicted futures in world coords
    target      : FloatTensor    [N, T, 2]  — ground truth in world coordinates
    metric_type : 'ade' | 'fde'

    Returns
    -------
    Scalar FloatTensor

    Implementation note
    -------------------
    We vectorise over K rather than looping.  For K=20 the loop is not a
    bottleneck today, but several upcoming metrics (AMD, AMV, collision rate)
    require pairwise operations across all K samples, and keeping everything
    in tensor form from the start makes those extensions straightforward.

    ADE vectorisation:
        preds            : [K, N, T, 2]
        target.unsqueeze : [1, N, T, 2]  (broadcasts across K)
        norm             : [K, N, T]
        .mean(dim=2)     : [K, N]        (average over time)
        .min(dim=0)      : [N]           (best sample per pedestrian)
        .mean()          : scalar

    FDE vectorisation:
        preds[:, :, -1, :]  : [K, N, 2]
        target[:, -1, :]    : [N, 2]  → unsqueeze → [1, N, 2]
        norm                : [K, N]
        .min(dim=0)         : [N]
        .mean()             : scalar
    """
    if metric_type == 'ade':
        # [K, N, T, 2] vs [1, N, T, 2] → [K, N, T] → [K, N]
        errors = torch.norm(
            preds - target.unsqueeze(0), p=2, dim=-1
        ).mean(dim=2)
    elif metric_type == 'fde':
        # [K, N, 2] vs [1, N, 2] → [K, N]
        errors = torch.norm(
            preds[:, :, -1, :] - target[:, -1, :].unsqueeze(0), p=2, dim=-1
        )
    else:
        raise ValueError(f"metric_type must be 'ade' or 'fde', got '{metric_type}'")

    # For each pedestrian, keep the error of the best sample: [N]
    min_errors, _ = errors.min(dim=0)

    return min_errors.mean()


def calculate_collision_rate(preds, threshold=0.1):
    """
    CR: Percentage of frames where any two agents are closer than 'threshold'.
    preds: [N, T, 2] (Absolute world coordinates)
    """
    N, T, _ = preds.shape
    if N < 2: return 0.0
    
    collisions = 0
    total_ped_frames = N * T
    
    for t in range(T):
        pos = preds[:, t, :] # [N, 2]
        dist_matrix = torch.cdist(pos, pos) # [N, N]
        # Ignore self-distance by adding large value to diagonal
        dist_matrix += torch.eye(N, device=pos.device) * 10
        collisions += (dist_matrix < threshold).sum().item()
    
    return collisions / (N * (N-1) * T) # Normalized by possible pairs


def calculate_psv(preds, radius=0.45):
    """
    PSV: Average number of pedestrians violating the social 'proxemic' bubble.
    """
    N, T, _ = preds.shape
    if N < 2: return 0.0
    
    violations = 0
    for t in range(T):
        dist_matrix = torch.cdist(preds[:, t, :], preds[:, t, :])
        dist_matrix += torch.eye(N, device=preds.device) * 10
        violations += (dist_matrix < radius).any(dim=1).sum().item()
        
    return violations / (N * T)


def calculate_distribution_metrics(preds_k):
    """
    AMD & AMV: Evaluate the uncertainty/spread of the K samples.
    preds_k: [K, N, T, 2]
    """
    K, N, T, _ = preds_k.shape
    # Flatten N and T to treat all predicted points as a distribution per sample
    # or calculate per timestep. Standard practice is per-timestep average.
    
    means = preds_k.mean(dim=0) # [N, T, 2]
    all_amd = []
    all_amv = []
    
    for n in range(N):
        for t in range(T):
            sample_points = preds_k[:, n, t, :] # [K, 2]
            diff = sample_points - means[n, t]
            cov = (diff.T @ diff) / (K - 1) + torch.eye(2, device=preds_k.device) * 1e-6
            
            # AMD: Mahalanobis distance of samples from their own mean
            inv_cov = torch.inverse(cov)
            md = torch.sqrt(torch.diag(diff @ inv_cov @ diff.T))
            all_amd.append(md.mean().item())
            
            # AMV: Max eigenvalue of the covariance (direction of max uncertainty)
            eigvals = torch.linalg.eigvalsh(cov)
            all_amv.append(eigvals.max().item())
            
    return np.mean(all_amd), np.mean(all_amv)

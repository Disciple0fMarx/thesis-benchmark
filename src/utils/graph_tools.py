import torch
import numpy as np


def build_adjacency_matrix(coords, threshold=5.0):
    """
    Args:
        coords: [N, 2] tensor of (x, y) coordinates for N pedestrians.
        threshold: distance in meters beyond which social influence is zero.
    Returns:
        A: [N, N] Adjacency matrix.
    """
    num_peds = coords.shape[0]
    if num_peds == 0:
        return torch.zeros((0, 0))

    # Calculate pairwise Euclidean distances
    # dists[i, j] = sqrt((xi-xj)^2 + (yi-yj)^2)
    dists = torch.cdist(coords, coords, p=2)

    # Calculate inverse distance for weights (adding epsilon to avoid div by zero)
    # We only apply this to the off-diagonal elements
    mask = (dists <= threshold) & (dists > 0)
    adj = torch.zeros_like(dists)
    adj[mask] = 1.0 / dists[mask]

    # For GNNs, we often add self-loops (the pedestrian's own motion history)
    adj += torch.eye(num_peds)
    
    return adj


def get_spatial_edges(batch_coords):
    """
    Computes graphs for a sequence.
    Args:
        batch_coords: [Num_Peds, Seq_Len, 2]
    Returns:
        List of Adjacency matrices, one per timestep.
    """
    num_peds, seq_len, _ = batch_coords.shape
    adj_sequences = []
    
    for t in range(seq_len):
        adj_t = build_adjacency_matrix(batch_coords[:, t, :])
        adj_sequences.append(adj_t)
        
    return torch.stack(adj_sequences) # [Seq_Len, Num_Peds, Num_Peds]

import os
import pandas as pd


def load_raw_obsmat(path):
    """
    Standard obsmat.txt format: 
    [frame_number, id, x, z, y, vx, vz, vy]
    We keep: frame, id, x, y
    """
    columns = ['frame', 'id', 'x', 'z', 'y', 'vx', 'vz', 'vy']
    df = pd.read_csv(path, sep='\s+', header=None, names=columns)
    return df[['frame', 'id', 'x', 'y']]


# Example usage for your 'ETH' chapter
# eth_data = load_raw_obsmat('data/raw/eth/obsmat.txt')


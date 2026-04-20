"""
---------------------------------------------------------------------------
Scene registry
---------------------------------------------------------------------------
Maps the canonical 5-fold benchmark name (used everywhere in the codebase,
in leave-one-out splits, in results tables, etc.) to its physical location
on disk AND its recording metadata.

frame_step: number of raw frames between consecutive trajectory samples
  ETH is recorded at 25 fps, sampled every 6th frame  → 0.4 s / step
  UCY is recorded at 25 fps, sampled every 10th frame → 0.4 s / step
  Both end up at the same effective rate (2.5 Hz), which is why the
  sequence lengths (obs=8, pred=12) correspond to 3.2 s and 4.8 s.

id_offset: a fixed, deterministic integer added to raw pedestrian IDs
  before combining scenes, so that ID 42 from 'hotel' and ID 42 from
  'zara1' are never confused.  Using hash() is NOT safe because Python
  randomises its hash seed across interpreter sessions.
---------------------------------------------------------------------------
"""

import os
import pandas as pd
import numpy as np


SCENE_REGISTRY = {
    #  canonical     dataset_dir  scene_dir   fps   frame_step  id_offset
    'eth':   dict(dataset='eth',  scene='univ',  fps=25, frame_step=6,  id_offset=0),
    'hotel': dict(dataset='eth',  scene='hotel', fps=25, frame_step=6,  id_offset=100_000),
    'univ':  dict(dataset='ucy',  scene='univ',  fps=25, frame_step=10, id_offset=200_000),
    'zara1': dict(dataset='ucy',  scene='zara1', fps=25, frame_step=10, id_offset=300_000),
    'zara2': dict(dataset='ucy',  scene='zara2', fps=25, frame_step=10, id_offset=400_000),
}
 
# All five canonical names in a fixed, reproducible order.
ALL_SCENES = ['eth', 'hotel', 'univ', 'zara1', 'zara2']


class TrajectoryLoader:
    """
    Standardised loader for the ETH-UCY pedestrian trajectory benchmark.
 
    The loader is intentionally thin: it only reads files from disk and
    returns plain Python / NumPy / Pandas objects.  No windowing, no
    normalisation, no tensor conversion — those belong in the generator
    and dataset classes respectively.
 
    Parameters
    ----------
    data_dir : str
        Path to the 'data/raw' directory that contains the 'eth/' and
        'ucy/' sub-directories.
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        

    def load_scene(self, canonical_name: str) -> dict:
        """
        Load a single scene by its canonical benchmark name.
 
        Parameters
        ----------
        canonical_name : str
            One of: 'eth', 'hotel', 'univ', 'zara1', 'zara2'.
 
        Returns
        -------
        dict with keys:
            'df'         : pd.DataFrame  – columns [frame, id, x, y]
            'H'          : np.ndarray | None  – 3×3 homography matrix
            'groups'     : list[list[int]]  – social group IDs (may be [])
            'fps'        : int  – recording frame rate
            'frame_step' : int  – raw frames between consecutive samples
            'name'       : str  – canonical name (echoed back for convenience)
        """
        if canonical_name not in SCENE_REGISTRY:
            raise ValueError(
                f"Unknown scene '{canonical_name}'. "
                f"Valid options: {list(SCENE_REGISTRY.keys())}"
            )
 
        meta = SCENE_REGISTRY[canonical_name]
        scene_dir = os.path.join(
            self.data_dir, meta['dataset'], meta['scene']
        )
        obsmat_path = os.path.join(scene_dir, 'obsmat.txt')
 
        if not os.path.exists(obsmat_path):
            raise FileNotFoundError(
                f"obsmat.txt not found at '{obsmat_path}'. "
                f"Make sure the raw data has been downloaded and placed under "
                f"'{self.data_dir}'."
            )
 
        df = self._load_obsmat(obsmat_path)
        H = self._load_h_matrix(scene_dir)
        groups = self._load_groups(scene_dir)
 
        return {
            'df':         df,
            'H':          H,
            'groups':     groups,
            'fps':        meta['fps'],
            'frame_step': meta['frame_step'],
            'name':       canonical_name,
        }

    def load_train_scenes(self, test_scene: str) -> list[dict]:
        """
        Load all scenes except *test_scene* for leave-one-out training.
 
        Each returned scene dict is identical to what `load_scene` returns,
        with one addition:
 
            'df_offset' : pd.DataFrame  – same as 'df' but with pedestrian
                          IDs shifted by the scene's fixed id_offset so that
                          IDs are globally unique when scenes are combined.
 
        Why return a *list* rather than a single concatenated DataFrame?
        Because each scene has its own frame_step (ETH uses 6, UCY uses 10).
        The sequence generator needs per-scene frame_step to check frame
        continuity correctly.  Concatenating frames from different scenes
        into one DataFrame and then trying to detect continuity is
        fundamentally broken.
 
        Parameters
        ----------
        test_scene : str
            Canonical name of the held-out test scene.
 
        Returns
        -------
        list[dict]  – one entry per training scene, in a fixed order.
        """
        if test_scene not in SCENE_REGISTRY:
            raise ValueError(
                f"Unknown test scene '{test_scene}'. "
                f"Valid options: {list(SCENE_REGISTRY.keys())}"
            )
 
        train_scenes = []
        for name in ALL_SCENES:
            if name == test_scene:
                continue
            scene = self.load_scene(name)
            # Add an ID-offset copy so callers can safely merge if needed,
            # but keep the original df intact as well.
            offset_df = scene['df'].copy()
            offset_df['id'] = offset_df['id'] + SCENE_REGISTRY[name]['id_offset']
            scene['df_offset'] = offset_df
            train_scenes.append(scene)
 
        return train_scenes

    def load_split_scene(self, canonical_name: str, train_ratio: float = 0.7) -> tuple[dict, dict]:
        """
        Loads a single scene and splits it chronologically into train/test sets.
        
        Returns
        -------
        (train_scene, test_scene) : tuple of dicts
        """
        scene = self.load_scene(canonical_name)
        df = scene['df']
        
        # Identify the unique frames and find the split point
        unique_frames = sorted(df['frame'].unique())
        split_idx = int(len(unique_frames) * train_ratio)
        split_frame = unique_frames[split_idx]
        
        # Split the DataFrame chronologically
        train_df = df[df['frame'] < split_frame].copy()
        test_df  = df[df['frame'] >= split_frame].copy()
        
        # Create two scene dicts that the Dataset/Generator can understand
        train_scene = scene.copy()
        train_scene['df'] = train_df
        train_scene['name'] = f"{canonical_name}_train"
        
        test_scene = scene.copy()
        test_scene['df'] = test_df
        test_scene['name'] = f"{canonical_name}_test"
        
        return train_scene, test_scene

    @staticmethod
    def _load_obsmat(path: str) -> pd.DataFrame:
        """
        Parse an obsmat.txt file into a clean DataFrame.
 
        obsmat.txt column layout (0-indexed):
            0: frame_number
            1: pedestrian_id
            2: pos_x   (world X, metres)
            3: pos_z   (world Z — vertical axis, not used)
            4: pos_y   (world Y, metres — the second horizontal axis)
            5: v_x
            6: v_z
            7: v_y
 
        We keep only frame, id, x, y.
        Coordinates are stored as float32 (sufficient precision for
        metre-scale pedestrian positions and much lighter than float64).
        """
        data = pd.read_csv(
            path,
            sep=r'\s+',
            header=None,
            usecols=[0, 1, 2, 4],  # frame, id, x, y
        )
        data.columns = ['frame', 'id', 'x', 'y']

        if data['x'].abs().max() > 100: 
            print(f"DEBUG: Large coordinates detected ({data['x'].max()}). Scaling to meters...")
            # If you don't have a working H-matrix, 0.05 is the standard 'fallback' 
            # for these datasets to bring pixels into a ~0-30m range.
            data['x'] = data['x'] * 0.05
            data['y'] = data['y'] * 0.05
            
        data = data.astype({
            'frame': np.int64,
            'id':    np.int64,
            'x':     np.float32,
            'y':     np.float32,
        })
        # Sort by frame then id so all downstream code can rely on ordering.
        data = data.sort_values(['frame', 'id']).reset_index(drop=True)
        return data

    @staticmethod
    def _load_h_matrix(scene_dir: str):
        """
        Load the 3×3 homography matrix (H.txt) if present.
        Returns None when the file is absent — callers must handle this.
        """
        h_path = os.path.join(scene_dir, 'H.txt')
        if os.path.exists(h_path):
            return np.loadtxt(h_path)
        return None

    @staticmethod
    def _load_groups(scene_dir: str) -> list[list[int]]:
        """
        Load social group annotations (groups.txt) if present.
 
        Format: one group per line, space-separated integer pedestrian IDs.
        Returns an empty list when the file is absent.
        """
        group_path = os.path.join(scene_dir, 'groups.txt')
        groups = []
        if os.path.exists(group_path):
            with open(group_path, 'r') as f:
                for line in f:
                    ids = [int(i) for i in line.split() if i.isdigit()]
                    if ids:
                        groups.append(ids)
        return groups

import torch
from torch.utils.data import Dataset
import numpy as np


class SocialDataset(Dataset):
    """
    PyTorch Dataset for ETH-UCY pedestrian trajectory prediction.

    Stores a flat list of social windows produced by SocialSequenceGenerator.
    Each window is one sample: all pedestrians present in a fixed-length
    time segment, with their observed and future trajectories.

    Coordinate conventions
    ----------------------
    The generator produces raw world coordinates.  This dataset applies one
    normalisation step in __getitem__:

        obs_norm[i, t] = obs[i, t] - origin[i]

    where origin[i] = obs[i, -1, :] is the last observed position of
    pedestrian i.  After this shift every pedestrian's coordinate frame is
    centred on their last seen position, which helps models generalise across
    scenes recorded at different world origins.

    IMPORTANT — what is NOT normalised:
        - 'pred' is kept in raw world coordinates.  Metrics (ADE, FDE) must
          be computed in world coordinates, so we never touch this array.
        - 'obs_rel' and 'pred_rel' are displacements (pos[t] - pos[t-1])
          and are already translation-invariant — no normalisation needed.

    Reconstructing absolute predictions
    ------------------------------------
    A model that outputs predictions in the normalised frame must convert back
    to world coordinates before metric computation.  Use the provided helper:

        pred_abs = SocialDataset.reconstruct_abs(pred_norm, origin)

    where pred_norm has shape [N_peds, pred_len, 2] and origin has shape
    [N_peds, 1, 2].  The helper broadcasts correctly.

    Batching
    --------
    Different windows contain different numbers of pedestrians, so windows
    cannot be naively stacked into a single tensor.  The provided
    social_collate function therefore returns LISTS of tensors, one entry
    per sample in the batch.  Every model's forward pass must loop over
    the batch list (or use scatter/pad operations) — do not assume the batch
    dimension is a regular tensor axis.

    Parameters
    ----------
    scenes : list[dict]
        One or more scene dicts as returned by TrajectoryLoader.load_scene()
        or TrajectoryLoader.load_train_scenes().  Each dict must have keys
        'df' and 'frame_step' at minimum.
    generator : SocialSequenceGenerator
        A configured generator instance.  The same generator is used for all
        scenes, which ensures consistent obs_len / pred_len / stride across
        the entire dataset.
    """

    def __init__(self, scenes: list[dict], generator):
        self.obs_len  = generator.obs_len
        self.pred_len = generator.pred_len

        self._obs:      list[np.ndarray] = []
        self._pred:     list[np.ndarray] = []
        self._obs_rel:  list[np.ndarray] = []
        self._pred_rel: list[np.ndarray] = []

        # Accept a single scene dict as a convenience (no need for the caller
        # to wrap it in a list when building a single-scene test dataset).
        if isinstance(scenes, dict):
            scenes = [scenes]

        for scene in scenes:
            windows = generator.generate_from_scene(scene)
            self._obs.extend(windows['obs'])
            self._pred.extend(windows['pred'])
            self._obs_rel.extend(windows['obs_rel'])
            self._pred_rel.extend(windows['pred_rel'])

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._obs)

    def __getitem__(self, idx: int) -> dict:
        """
        Return one social window as a dict of tensors.

        Returns
        -------
        dict with keys:
            'obs'      : FloatTensor [N_peds, obs_len,  2]  — normalised absolute
            'pred'     : FloatTensor [N_peds, pred_len, 2]  — RAW world coords
            'obs_rel'  : FloatTensor [N_peds, obs_len,  2]  — displacements (input)
            'pred_rel' : FloatTensor [N_peds, pred_len, 2]  — displacements (target)
            'origin'   : FloatTensor [N_peds, 1,        2]  — last obs position
                         in world coords; add this to any normalised prediction
                         to get back to world coords for metric computation.

        The N_peds dimension varies between samples.  See social_collate.
        """
        obs      = torch.from_numpy(self._obs[idx])       # [N, obs_len, 2]
        pred     = torch.from_numpy(self._pred[idx])      # [N, pred_len, 2]
        obs_rel  = torch.from_numpy(self._obs_rel[idx])   # [N, obs_len, 2]
        pred_rel = torch.from_numpy(self._pred_rel[idx])  # [N, pred_len, 2]

        # origin: last observed position per pedestrian — [N, 1, 2]
        # Shape [N, 1, 2] (not [N, 2]) so that it broadcasts directly against
        # [N, pred_len, 2] predictions without any unsqueeze calls in model code.
        origin = obs[:, -1:, :].clone()   # [N, 1, 2]

        # Shift absolute observations so that every pedestrian's last seen
        # position becomes the origin.  obs_rel is already translation-
        # invariant so it does not need this treatment.
        obs_norm = obs - origin           # [N, obs_len, 2]

        return {
            'obs':      obs_norm,   # normalised: centred on last obs position
            'pred':     pred,       # raw world coords — for metric computation
            'obs_rel':  obs_rel,    # displacements — for model input
            'pred_rel': pred_rel,   # displacements — for model loss
            'origin':   origin,     # world position of the normalisation origin
        }

    # ------------------------------------------------------------------
    # Coordinate reconstruction helper
    # ------------------------------------------------------------------

    @staticmethod
    def reconstruct_abs(
        pred_norm: torch.Tensor,
        origin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert normalised predictions back to world coordinates.

        This must be called before computing ADE / FDE, which are defined
        in world coordinates.

        Parameters
        ----------
        pred_norm : FloatTensor [..., pred_len, 2]
            Predictions in the normalised (origin-centred) frame.
            The leading dimensions can be anything — e.g. [N, pred_len, 2]
            for a single sample, or [K, N, pred_len, 2] for K stochastic
            samples from a generative model.
        origin : FloatTensor [N, 1, 2]
            The origin tensor from the dataset item, as returned by
            __getitem__.  Broadcasting handles the leading K dimension
            automatically when pred_norm has shape [K, N, pred_len, 2].

        Returns
        -------
        FloatTensor  — same shape as pred_norm, in world coordinates.

        Example
        -------
        # Single deterministic prediction
        pred_abs = SocialDataset.reconstruct_abs(pred_norm, batch['origin'])

        # K stochastic samples from a generative model (e.g. MoFlow)
        # pred_samples: [K, N, pred_len, 2]
        # origin:       [N, 1, 2]
        pred_abs = SocialDataset.reconstruct_abs(pred_samples, origin)
        # Result: [K, N, pred_len, 2] — each of the K samples in world coords
        """
        return pred_norm + origin

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SocialDataset("
            f"windows={len(self)}, "
            f"obs_len={self.obs_len}, "
            f"pred_len={self.pred_len})"
        )


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def social_collate(batch: list[dict]) -> dict:
    """
    Collate function for DataLoader when using SocialDataset.

    IMPORTANT: This intentionally returns LISTS of tensors rather than a
    single stacked tensor.  This is necessary because each sample contains
    a different number of pedestrians (N_peds), making a naive torch.stack
    impossible.

    Every model's forward pass must iterate over the batch list.  For
    example:

        for obs, pred, obs_rel, pred_rel, origin in zip(
            batch['obs'], batch['pred'],
            batch['obs_rel'], batch['pred_rel'],
            batch['origin'],
        ):
            # obs:  [N_peds, obs_len, 2]  — varies per sample
            # pred: [N_peds, pred_len, 2]
            out = model(obs, obs_rel)
            loss += criterion(out, pred_rel)

    Parameters
    ----------
    batch : list[dict]
        List of dicts as returned by SocialDataset.__getitem__.

    Returns
    -------
    dict with the same keys as __getitem__, where every value is a list
    of tensors (one per sample in the batch) instead of a single tensor.
    """
    return {
        'obs':      [item['obs']      for item in batch],
        'pred':     [item['pred']     for item in batch],
        'obs_rel':  [item['obs_rel']  for item in batch],
        'pred_rel': [item['pred_rel'] for item in batch],
        'origin':   [item['origin']   for item in batch],
    }

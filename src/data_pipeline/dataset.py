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

    Optional: TrajectoryNormaliser
    ------------------------------
    Some models (e.g. MoFlow) require future trajectories scaled to [-1, 1].
    Attach a fitted TrajectoryNormaliser via set_normaliser() to make
    'pred_norm' available in every batch item.
 
    CRITICAL — leave-one-out correctness:
        The normaliser MUST be fit on the TRAINING fold only, then the same
        fitted normaliser must be passed to BOTH the training and test
        datasets.  The test dataset must never compute its own statistics.
 
            train_dataset.set_normaliser(normaliser)   # training bounds
            test_dataset.set_normaliser(normaliser)    # SAME training bounds
 
    When a normaliser is set, __getitem__ returns an additional key:
        'pred_norm' : FloatTensor [N_peds, pred_len, 2]
                      Future trajectory normalised to [-1, 1] using training
                      fold statistics.  'pred' (raw world coords) is always
                      present alongside it for metric computation.
 
    Models that do not need normalisation simply ignore 'pred_norm'.

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

        # Optional TrajectoryNormaliser — None means 'pred_norm' will not
        # appear in batch items.  Set via set_normaliser() after construction.
        self._normaliser = None

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
    # Normaliser attachment
    # ------------------------------------------------------------------
 
    def set_normaliser(self, normaliser) -> None:
        """
        Attach a fitted TrajectoryNormaliser to this dataset.
 
        Once set, every item returned by __getitem__ will include a
        'pred_norm' key containing the min-max normalised future trajectory.
 
        Parameters
        ----------
        normaliser : TrajectoryNormaliser
            Must already be fitted (normaliser.is_fitted == True).
            Always pass the normaliser fitted on the TRAINING fold — even
            when calling this on the test dataset.  The test dataset must
            use training-fold bounds for a fair benchmark.
 
        Raises
        ------
        RuntimeError  if the normaliser has not been fitted yet.
        """
        if not normaliser.is_fitted:
            raise RuntimeError(
                "The normaliser has not been fitted yet. "
                "Call normaliser.fit(train_dataset) before set_normaliser()."
            )
        self._normaliser = normaliser

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

        Present only when a normaliser has been attached via set_normaliser():
            'pred_norm': FloatTensor [N_peds, pred_len, 2]  — future in [-1, 1]
                         Uses training-fold statistics exclusively.
                         NEVER use for metric computation — use 'pred' instead.

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

        item = {
            'obs':      obs_norm,   # normalised: centred on last obs position
            'pred':     pred,       # raw world coords — for metric computation
            'obs_rel':  obs_rel,    # displacements — for model input
            'pred_rel': pred_rel,   # displacements — for model loss
            'origin':   origin,     # world position of the normalisation origin
        }

        if self._normaliser is not None:
            item['pred_norm'] = self._normaliser.transform(pred - origin)
 
        return item

    # ------------------------------------------------------------------
    # Coordinate reconstruction helper
    # ------------------------------------------------------------------

    @staticmethod
    def reconstruct_abs(
        pred_centred: torch.Tensor,
        origin:       torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert origin-centred predictions back to world coordinates.
 
        Must be called before computing ADE / FDE, which are defined in
        world coordinates.
 
        Parameters
        ----------
        pred_centred : FloatTensor [..., pred_len, 2]
            Predictions in the origin-centred frame.  Leading dimensions
            are arbitrary — works for [N, T, 2] and [K, N, T, 2] alike.
        origin : FloatTensor [N, 1, 2]
            The origin tensor from the dataset item.  Broadcasting handles
            the leading K dimension automatically.
 
        Returns
        -------
        FloatTensor — same shape as pred_centred, in world coordinates.
        """
        return pred_centred + origin

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        norm_str = (
            f", normaliser={self._normaliser.mode}"
            if self._normaliser is not None else ""
        )
        return (
            f"SocialDataset("
            f"windows={len(self)}, "
            f"obs_len={self.obs_len}, "
            f"pred_len={self.pred_len}"
            f"{norm_str})"
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
    result = {
        'obs':      [item['obs']      for item in batch],
        'pred':     [item['pred']     for item in batch],
        'obs_rel':  [item['obs_rel']  for item in batch],
        'pred_rel': [item['pred_rel'] for item in batch],
        'origin':   [item['origin']   for item in batch],
    }

    if 'pred_norm' in batch[0]:
        result['pred_norm'] = [item['pred_norm'] for item in batch]
 
    return result

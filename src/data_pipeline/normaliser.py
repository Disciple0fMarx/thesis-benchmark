import os
import numpy as np
import torch


class TrajectoryNormaliser:
    """
    Fits coordinate normalisation statistics on a SocialDataset training fold
    and provides transform / inverse_transform for future trajectories.

    Only future trajectories ('pred') are normalised.  Observed trajectories
    ('obs') are already origin-centred by SocialDataset and do not need
    further normalisation.  Displacement tensors ('obs_rel', 'pred_rel') are
    translation-invariant by construction and are never normalised.

    Parameters
    ----------
    mode : str
        Normalisation mode.  Currently only 'minmax' is supported.

    Usage (one fold of leave-one-out)
    ----------------------------------
        normaliser = TrajectoryNormaliser(mode='minmax')
        normaliser.fit(train_dataset)           # scans training fold only

        train_dataset.set_normaliser(normaliser)
        test_dataset.set_normaliser(normaliser)  # training bounds on test set

        trainer.save_checkpoint(path)           # saves model + normaliser
        # ... later ...
        trainer.load_checkpoint(path)           # restores both
        test_dataset.set_normaliser(trainer.normaliser)
    """

    SUPPORTED_MODES = ('minmax', 'standard')

    def __init__(self, mode: str = 'standard'):
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unknown normalisation mode '{mode}'. "
                f"Supported: {self.SUPPORTED_MODES}"
            )
        self.mode    = mode
        self._fitted = False

        # Per-coordinate bounds, shape [2] each (x and y).
        # Stored as numpy arrays internally; converted to tensors on demand.
        self._min: np.ndarray | None = None   # [2]
        self._max: np.ndarray | None = None   # [2]

        self._mean = None
        self._std  = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, dataset) -> 'TrajectoryNormaliser':
        """
        Compute normalisation statistics from a SocialDataset.

        Scans the raw '_pred' arrays directly (no DataLoader, no GPU).
        This is an O(N * T) pass over numpy arrays — fast even for large
        datasets.

        Parameters
        ----------
        dataset : SocialDataset
            Must have a '_pred' attribute (list of np.ndarray [N_i, T, 2]).
            Fits on whatever data is in this dataset — always pass the
            TRAINING fold, never the test fold.

        Returns
        -------
        self  (for method chaining)
        """
        if not hasattr(dataset, '_pred'):
            raise AttributeError(
                "dataset must be a SocialDataset instance with a '_pred' "
                "attribute.  Did you pass the right object?"
            )

        if len(dataset._pred) == 0:
            raise ValueError("Cannot fit on an empty dataset.")

        # Stack all future positions into one array for efficient min/max.
        # Each element is [N_i, T, 2]; stack along the first axis after
        # reshaping to [N_i * T, 2] so we get [total_positions, 2].
        all_coords = np.concatenate(
            [(p - dataset._obs[i][:, -1:, :]).reshape(-1, 2)
            for i, p in enumerate(dataset._pred)], axis=0
        )                                                  # [M, 2]

        if self.mode == 'standard':
            self._mean = np.mean(all_coords, axis=(0, 1))
            self._std = np.std(all_coords, axis=(0, 1))
        elif self.mode == 'minmax':
            # self._min = all_coords.min(axis=0)             # [2]
            # self._max = all_coords.max(axis=0)             # [2]

            # Use the absolute maximum displacement to create a symmetric boundary
            # This ensures that a predicted 0.0 stays exactly at the last observed position.
            abs_max = np.abs(all_coords).max(axis=0)
            self._max = abs_max
            self._min = -abs_max

            # Guard against degenerate cases (all identical coordinates).
            # range_ = self._max - self._min
            if np.any(self._max < 1e-6):
                raise ValueError(
                    f"Degenerate coordinate range detected: min={self._min}, "
                    f"max={self._max}.  Check the training data."
                )

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Transform / inverse
    # ------------------------------------------------------------------

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalise a future trajectory tensor.

        Parameters
        ----------
        x : FloatTensor [..., 2]
            Future positions in any coordinate frame.  The last dimension
            must be 2 (x, y).  Leading dimensions are arbitrary — this
            works for [N, T, 2], [K, N, T, 2], etc.

        Returns
        -------
        FloatTensor same shape as x, values in [-1, 1].
        """
        self._check_fitted()

        if self.mode == 'standard':
            # Use mean/std tensors
            mu, sigma = self._stats_as_tensors(x.device)
            return (x - mu) / (sigma + 1e-6)
        else:
            x_min, x_max = self._bounds_as_tensors(x.device)

        # if self.mode == 'minmax':
            denom = (x_max - x_min).clamp(min=1e-6)
            return (x - x_min) / denom * 2.0 - 1.0

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denormalise a normalised future trajectory tensor.

        Parameters
        ----------
        x : FloatTensor [..., 2]
            Normalised positions in [-1, 1].

        Returns
        -------
        FloatTensor same shape as x, in original coordinate units.
        """
        self._check_fitted()

        if self.mode == 'standard':
            mu, sigma = self._stats_as_tensors(x.device)
            return x * sigma + mu
        else:
            x_min, x_max = self._bounds_as_tensors(x.device)

        # if self.mode == 'minmax':
            denom = (x_max - x_min).clamp(min=1e-6)
            return (x + 1.0) / 2.0 * denom + x_min

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save normalisation statistics to disk.

        The file is a small dict — not a full checkpoint.  It is meant to
        be embedded inside model checkpoints via:

            torch.save({'model': ..., 'normaliser': normaliser.state_dict()}, path)

        But it can also be saved standalone for inspection.
        """
        self._check_fitted()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.save(path, self.state_dict(), allow_pickle=True)

    def load(self, path: str) -> 'TrajectoryNormaliser':
        """
        Load normalisation statistics from a standalone .npy file.
        For loading from a model checkpoint use load_state_dict() instead.
        """
        state = np.load(path, allow_pickle=True).item()
        self.load_state_dict(state)
        return self

    def state_dict(self) -> dict:
        """
        Return a plain dict suitable for embedding in torch.save checkpoints.

        Bounds are stored as Python lists (not numpy arrays) so the checkpoint
        can be loaded with weights_only=True in PyTorch 2.6+, which rejects
        numpy globals by default.
        """
        self._check_fitted()
        return {
            'mode': self.mode,
            'min': self._min.tolist() if self._min is not None else None,
            'max': self._max.tolist() if self._max is not None else None,
            'mean': self._mean.tolist() if self._mean is not None else None,
            'std': self._std.tolist() if self._std is not None else None,
            # 'min':  self._min.tolist(),
            # 'max':  self._max.tolist(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore from a state dict returned by state_dict()."""
        if state['mode'] != self.mode:
            raise ValueError(
                f"Mode mismatch: normaliser is '{self.mode}' but "
                f"checkpoint has '{state['mode']}'."
            )
        self._min    = np.array(state['min'],  dtype=np.float32)
        self._max    = np.array(state['max'],  dtype=np.float32)
        self._fitted = True

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def __repr__(self) -> str:
        if not self._fitted:
            return f"TrajectoryNormaliser(mode='{self.mode}', fitted=False)"
        return (
            f"TrajectoryNormaliser(mode='{self.mode}', "
            f"min={self._min}, max={self._max})"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "TrajectoryNormaliser has not been fitted yet. "
                "Call fit(train_dataset) before transform() or save()."
            )

    def _bounds_as_tensors(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return min and max as float32 tensors on the given device."""
        x_min = torch.tensor(self._min, dtype=torch.float32, device=device)
        x_max = torch.tensor(self._max, dtype=torch.float32, device=device)
        return x_min, x_max

    def _stats_as_tensors(self, device):
        """Helper for standard mode"""
        mu = torch.tensor(self._mean, dtype=torch.float32, device=device)
        sigma = torch.tensor(self._std, dtype=torch.float32, device=device)
        return mu, sigma

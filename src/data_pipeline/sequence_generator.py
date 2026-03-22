import numpy as np
import torch


class SocialSequenceGenerator:
    """
    Converts a scene DataFrame (as produced by TrajectoryLoader) into a list
    of fixed-length social windows ready for model training and evaluation.
 
    Each window contains all pedestrians that are *continuously present* for
    the full obs_len + pred_len frames, so every model sees complete
    trajectories with no missing timestamps.
 
    Parameters
    ----------
    obs_len : int
        Number of observed (input) frames. Default 8 → 3.2 s at 2.5 Hz.
    pred_len : int
        Number of future (target) frames. Default 12 → 4.8 s at 2.5 Hz.
    stride : int or None
        Number of frames to advance the window on each step.
        None (default) → use pred_len, which gives non-overlapping prediction
        windows.  This is the convention used by Social-GAN, STGCNN, and the
        LED / MoFlow papers and is required for a fair benchmark.
        Set to 1 only if you specifically need dense windowing (e.g. for
        visualisation), and be aware that it creates heavily correlated windows.
    min_peds : int
        Minimum number of pedestrians required per window.  Default 2 because
        social models need at least two agents to model interaction.  Set to 1
        only for ablation studies on non-social baselines.
    """

    def __init__(
        self,
        obs_len: int = 8,
        pred_len: int = 12,
        stride: int | None = None,
        min_peds: int = 2,
    ):
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        # Non-overlapping prediction windows by default — matches benchmark papers.
        self.stride = stride if stride is not None else pred_len
        self.min_peds = min_peds

    def generate_from_scene(self, scene: dict) -> dict:
        """
        Convenience wrapper that accepts a scene dict as returned by
        TrajectoryLoader.load_scene() and forwards to generate().
 
        Parameters
        ----------
        scene : dict
            Must have at minimum keys 'df' and 'frame_step'.
 
        Returns
        -------
        Same dict as generate().
        """
        return self.generate(scene['df'], scene['frame_step'])

    def generate(self, df, frame_step: int) -> dict:
        """
        Process a scene DataFrame into social sequence windows.
 
        Parameters
        ----------
        df : pd.DataFrame
            Columns: [frame, id, x, y].  Must be sorted by frame then id
            (TrajectoryLoader guarantees this).
        frame_step : int
            The expected gap between consecutive frame numbers in this scene.
            ETH scenes use 6; UCY scenes use 10.  Passed explicitly so that
            the generator is stateless and safe to call for multiple scenes
            with different metadata.
 
        Returns
        -------
        dict with keys:
            'obs'      : list of np.ndarray, each [N_peds, obs_len,  2], float32
            'pred'     : list of np.ndarray, each [N_peds, pred_len, 2], float32
            'obs_rel'  : list of np.ndarray, each [N_peds, obs_len,  2], float32
            'pred_rel' : list of np.ndarray, each [N_peds, pred_len, 2], float32
 
        Shapes
        ------
        - The first dimension (N_peds) varies between windows.
        - Coordinates are in the original world-coordinate space (metres for
          ETH-UCY).  No normalisation is applied here — that belongs in the
          Dataset class.
        - 'obs_rel' and 'pred_rel' are frame-to-frame displacements:
            rel[t] = pos[t] - pos[t-1]  for t > 0
            rel[0] = [0.0, 0.0]          (zero displacement at the first step)
          This matches the convention used in Social-GAN, Social-STGCNN, LED,
          and MoFlow.  DO NOT use any other value at index 0 — it would produce
          sequences that differ from all baseline implementations and break
          benchmark comparability.
 
        Notes on frame continuity
        -------------------------
        A window is only accepted if ALL consecutive frame pairs in the window
        are exactly frame_step apart.  Checking only the (first, last) pair is
        insufficient: a gap in the middle would pass that check but produce
        sequences where two supposed "adjacent" time steps are actually 2×
        frame_step apart — effectively a time-warp in the middle of the window.
        """
        all_obs:      list[np.ndarray] = []
        all_pred:     list[np.ndarray] = []
        all_obs_rel:  list[np.ndarray] = []
        all_pred_rel: list[np.ndarray] = []
 
        frames = sorted(df['frame'].unique())
        n_frames = len(frames)
 
        # Build a frame→index lookup for O(1) slicing.
        frame_to_idx = {f: i for i, f in enumerate(frames)}
 
        # Slide the window.
        i = 0
        while i + self.seq_len <= n_frames:
            window_frames = frames[i : i + self.seq_len]
 
            # --- Frame continuity check (consecutive differences) -----------
            # We need every adjacent pair to be exactly frame_step apart.
            # Using only (last - first) would miss gaps in the middle.
            diffs = np.diff(window_frames)
            if not np.all(diffs == frame_step):
                # The window contains a gap.  Advance past it by finding the
                # first broken pair, then jumping to start a new window from
                # the frame immediately after the gap.
                bad_pos = int(np.argmax(diffs != frame_step))
                # bad_pos is the index within window_frames where the gap
                # starts.  The gap is between window_frames[bad_pos] and
                # window_frames[bad_pos+1].  We want i such that
                # window_frames[bad_pos+1] becomes the new window start.
                i += bad_pos + 1
                continue
 
            # --- Filter pedestrians present in every frame ------------------
            window_df = df[df['frame'].isin(window_frames)]
            counts = window_df.groupby('id')['frame'].count()
            valid_ids = counts[counts == self.seq_len].index.tolist()
 
            if len(valid_ids) < self.min_peds:
                i += self.stride
                continue
 
            # --- Extract absolute trajectories in a stable order ------------
            # Sorting by id ensures that the pedestrian order is deterministic
            # across windows, which matters when debugging and for any model
            # that relies on consistent agent indexing across time.
            valid_ids_sorted = sorted(valid_ids)
 
            peds_traj = []
            for ped_id in valid_ids_sorted:
                traj = (
                    window_df[window_df['id'] == ped_id]
                    .sort_values('frame')[['x', 'y']]
                    .values.astype(np.float32)
                )
                peds_traj.append(traj)
 
            # Stack to [N_peds, seq_len, 2]
            trajs = np.stack(peds_traj)  # (N, seq_len, 2)
 
            # --- Compute relative (displacement) sequences ------------------
            # rel[t] = pos[t] - pos[t-1].
            # At t=0 there is no previous frame, so we use [0, 0].
            # This is the standard convention; do not change it.
            trajs_rel = np.zeros_like(trajs)              # (N, seq_len, 2)
            trajs_rel[:, 1:, :] = trajs[:, 1:, :] - trajs[:, :-1, :]
            # trajs_rel[:, 0, :] is already 0.0 from np.zeros_like.
 
            # --- Split obs / pred -------------------------------------------
            all_obs.append(trajs[:, :self.obs_len, :])
            all_pred.append(trajs[:, self.obs_len:, :])
            all_obs_rel.append(trajs_rel[:, :self.obs_len, :])
            all_pred_rel.append(trajs_rel[:, self.obs_len:, :])
 
            i += self.stride
 
        return {
            'obs':      all_obs,
            'pred':     all_pred,
            'obs_rel':  all_obs_rel,
            'pred_rel': all_pred_rel,
        }

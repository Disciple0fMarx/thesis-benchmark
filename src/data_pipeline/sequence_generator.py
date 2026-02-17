import numpy as np
import torch


class SocialSequenceGenerator:
    def __init__(self, obs_len=8, pred_len=12, frame_step=None):
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        # frame_step should be 6 for ETH and 10 for UCY to get 0.4s
        self.frame_step = frame_step 

    def generate(self, df):
        """
        Processes a DataFrame into social sequences.
        Returns:
            obs: list of [Num_Peds, Obs_Len, 2]
            pred: list of [Num_Peds, Pred_Len, 2]
            obs_rel: list of [Num_Peds, Obs_Len, 2] (offsets)
            pred_rel: list of [Num_Peds, Pred_Len, 2] (offsets)
        """
        all_obs, all_pred = [], []
        all_obs_rel, all_pred_rel = [], []

        # 1. Group by frame to easily slide windows
        frames = sorted(df['frame'].unique())
        
        for i in range(0, len(frames) - self.seq_len + 1):
            frame_window = frames[i : i + self.seq_len]
            
            # CHECK: Ensure frames are evenly spaced (e.g., all 10 frames apart)
            # This prevents "time-warping" if some frames are missing in the raw data
            if self.frame_step:
                if (frame_window[-1] - frame_window[0]) != (self.seq_len - 1) * self.frame_step:
                    continue

            window_df = df[df['frame'].isin(frame_window)]
            
            # 2. Filter for pedestrians present in EVERY frame of this window
            counts = window_df['id'].value_counts()
            valid_peds = counts[counts == self.seq_len].index.tolist()
            
            if len(valid_peds) < 2: # 'Social' models need at least 2 people to interact
                continue

            # 3. Extract and Calculate Velocities/Offsets
            peds_obs, peds_pred = [], []
            peds_obs_rel, peds_pred_rel = [], []

            for ped_id in valid_peds:
                traj = window_df[window_df['id'] == ped_id].sort_values(by='frame')[['x', 'y']].values
                
                # Absolute coordinates
                obs_part = traj[:self.obs_len]
                pred_part = traj[self.obs_len:]
                
                # Relative coordinates (Displacements: pos_t - pos_{t-1})
                # We pad the first relative coord with (0,0) or the first actual movement
                traj_rel = np.zeros_like(traj)
                traj_rel[1:] = traj[1:] - traj[:-1]
                traj_rel[0] = traj_rel[1] # Simple smoothing for the first velocity point
                
                peds_obs.append(obs_part)
                peds_pred.append(pred_part)
                peds_obs_rel.append(traj_rel[:self.obs_len])
                peds_pred_rel.append(traj_rel[self.obs_len:])

            # Convert to arrays and store
            all_obs.append(np.stack(peds_obs))
            all_pred.append(np.stack(peds_pred))
            all_obs_rel.append(np.stack(peds_obs_rel))
            all_pred_rel.append(np.stack(peds_pred_rel))

        return all_obs, all_pred, all_obs_rel, all_pred_rel

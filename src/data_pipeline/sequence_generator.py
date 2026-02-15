import numpy as np
import torch


class SocialSequenceGenerator:
    def __init__(self, obs_len=8, pred_len=12, step=1):
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.step = step

    def generate(self, df):
        """
        Processes a DataFrame from TrajectoryLoader into sequences.
        """
        # Ensure data is sorted by frame for sliding window
        df = df.sort_values(by='frame')
        frames = sorted(df['frame'].unique())
        
        all_sequences = []
        
        # Sliding window across frames
        for i in range(0, len(frames) - self.seq_len + 1, self.step):
            frame_window = frames[i : i + self.seq_len]
            window_df = df[df['frame'].isin(frame_window)]
            
            # Identify pedestrians present for the ENTIRE duration (8+12 frames)
            # This is the standard benchmark requirement.
            counts = window_df['id'].value_counts()
            valid_peds = counts[counts == self.seq_len].index.tolist()
            
            if len(valid_peds) == 0:
                continue

            # Extract coordinates for valid pedestrians
            # Resulting shape: [Num_Valid_Peds, Seq_Len, 2]
            peds_data = []
            for ped_id in valid_peds:
                ped_df = window_df[window_df['id'] == ped_id].sort_values(by='frame')
                peds_data.append(ped_df[['x', 'y']].values)
            
            all_sequences.append(np.stack(peds_data))
            
        return all_sequences

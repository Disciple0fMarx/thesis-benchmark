from src.data_pipeline.loader import TrajectoryLoader
from src.data_pipeline.sequence_generator import SocialSequenceGenerator
from src.utils.viz import plot_scene

import yaml
import torch


# 1. Load Config
with open('configs/data_config.yml', 'r') as f:
    config = yaml.safe_load(f)

# 2. Extract Data using YOUR loader
loader = TrajectoryLoader(config['data']['raw_path'])
scene_data = loader.load_scene('ucy', 'univ')
raw_df = scene_data['df']

# ETH (15fps) needs step 6 | UCY (25fps) needs step 10
f_step = 6 if scene_data['fps'] == 15 else 10

# 3. Pass to Generator
gen = SocialSequenceGenerator(
    obs_len=config['data']['t_obs'], 
    pred_len=config['data']['t_pred'],
    frame_step=f_step
    # step=config['data']['sliding_step']
)
sequences = gen.generate(raw_df)

print(f"Loaded {len(sequences)} windows from UCY Univ.")
print(f"Each window contains [Peds, {config['data']['t_obs'] + config['data']['t_pred']}, 2] coordinates.")

# Use the H-Matrix for visualization later
h_matrix = scene_data['H']

# Use the groups to color-code your plots
groups = scene_data['groups']


# Unpack all 4 outputs (Absolute and Relative)
obs, pred, obs_rel, pred_rel = sequences

# 4. Prepare for Visualization
# Convert the first window to Tensors to mimic the Dataset output
# We wrap them in a list because viz.py expects a 'batch' format
obs_tensor_list = [torch.from_numpy(obs[0]).float()]
pred_tensor_list = [torch.from_numpy(pred[0]).float()]

print(f"Total windows generated: {len(obs)}")
print(f"Visualizing window 0 with {obs[0].shape[0]} pedestrians...")

# 5. Run Plot
plot_scene(
    obs=obs_tensor_list, 
    pred=pred_tensor_list, 
    title=f"UCY Univ - Window 0 (Step: {f_step})",
    save_path="results/viz_debug_ucy.png"
)

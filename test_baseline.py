import yaml
import torch
from torch.utils.data import DataLoader

# Import our custom modules
from src.data_pipeline.loader import TrajectoryLoader
from src.data_pipeline.sequence_generator import SocialSequenceGenerator
from src.data_pipeline.dataset import SocialDataset, social_collate
from src.utils.viz import plot_scene
from src.models.baselines import ConstantVelocityModel
from src.evaluation.evaluator import Evaluator

# 1. Load Configuration
print("Loading config...")
with open('configs/data_config.yml', 'r') as f:
    config = yaml.safe_load(f)

# 2. Initialize Pipeline Components
print("Initializing pipeline...")
loader = TrajectoryLoader(config['data']['raw_path'])
generator = SocialSequenceGenerator(
    obs_len=config['data']['t_obs'],
    pred_len=config['data']['t_pred'],
    step=config['data']['sliding_step']
)

# 3. Create Dataset (Let's just load ETH Univ for a quick test)
print("Creating dataset (this might take a moment)...")
train_scenes = [
    # ('eth', 'univ'),
    # ('eth', 'hotel'),
    ('ucy', 'univ'),
]
dataset = SocialDataset(train_scenes, loader, generator, config)

# 4. Create DataLoader with our Custom Collator
# batch_size=4 means we load 4 distinct scenes
train_loader = DataLoader(
    dataset, 
    batch_size=4, 
    shuffle=True,
    collate_fn=social_collate
)

# 5. Get one batch
print("Fetching one batch...")
batch = next(iter(train_loader))
obs_list, target_list, adj_list = batch

print(f"Batch received. Contains {len(obs_list)} scenes.")
print(f"Scene 0 peds: {obs_list[0].shape[0]}, Scene 1 peds: {obs_list[1].shape[0]}")

# 6. Run Baseline Model
print("\nRunning Constant Velocity Baseline...")
cvm = ConstantVelocityModel(pred_len=config['data']['t_pred'])
# We don't need gradients for a baseline test
with torch.no_grad():
    pred_list = cvm(obs_list)

print(f"Prediction shape for Scene 0: {pred_list[0].shape}")
# Expected: [Num_Peds_in_Scene0, 12, 2]

# 7. Visualize results
print("\nVisualizing Scene 0 (Ground Truth)...")
# Plot Ground Truth
plot_scene(
    obs_list,
    target_list,
    adj_list,
    sample_idx=0,
    title="Ground Truth & Social Graph",
    save_path="results/viz_gt_ucy.png"
)

print("Visualizing Scene 0 (CVM Prediction)...")
# Plot CVM Prediction (Pass CVM output as the 'pred' argument)
# Note: We pass None for adj because CVM doesn't use the graph
plot_scene(
    obs_list,
    pred_list,
    adj=None,
    sample_idx=0,
    title="Constant Velocity Prediction",
    save_path="results/viz_cvm_ucy.png"
)

evaluator = Evaluator(cvm, config)
results = evaluator.evaluate(train_loader)

print("\n📊 ETH Univ Benchmark Results (CVM):")
print(f"ADE: {results['ADE']:.4f} meters")
print(f"FDE: {results['FDE']:.4f} meters")

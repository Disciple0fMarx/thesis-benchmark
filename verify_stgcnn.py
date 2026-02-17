import torch
import yaml
from torch.utils.data import DataLoader
from src.data_pipeline.loader import TrajectoryLoader
from src.data_pipeline.sequence_generator import SocialSequenceGenerator
from src.data_pipeline.dataset import SocialDataset, social_collate
from src.models.stgcnn import STGCNN
from src.evaluation.evaluator import Evaluator

# 1. Load config and data
with open('configs/data_config.yml', 'r') as f:
    config = yaml.safe_load(f)

loader = TrajectoryLoader(config['data']['raw_path'])
generator = SocialSequenceGenerator(obs_len=8, pred_len=12)
test_dataset = SocialDataset([('eth', 'univ')], loader, generator, config)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=social_collate)

# 2. Load the trained model
model = STGCNN(obs_len=8, pred_len=12)
model.load_state_dict(torch.load('results/checkpoints/stgcnn_hotel.pth', weights_only=True))

# 3. Evaluate
evaluator = Evaluator(model, config)
results = evaluator.evaluate(test_loader)

print("\n🔥 STGCNN Benchmark Results:")
print(f"ADE: {results['ADE']:.4f} meters")
print(f"FDE: {results['FDE']:.4f} meters")

import torch
import yaml
# from src.models.social_lstm import SocialLSTM
from src.models.stgcnn import STGCNN
from src.data_pipeline.loader import TrajectoryLoader
from src.data_pipeline.sequence_generator import SocialSequenceGenerator
from src.data_pipeline.dataset import SocialDataset
from src.utils.viz_predictions import visualize_model_results

# Load Config and Setup
with open('configs/data_config.yml', 'r') as f:
    config = yaml.safe_load(f)

loader = TrajectoryLoader(config['data']['raw_path'])
generator = SocialSequenceGenerator(obs_len=8, pred_len=12)
val_dataset = SocialDataset([('ucy', 'zara1')], loader, generator, config)

# Load Model
model = STGCNN(obs_len=8, pred_len=12)
checkpoint = torch.load("results/checkpoints/stgcnn_best_zara1.pth")
model.load_state_dict(checkpoint['state_dict'])

# Run Viz
visualize_model_results(model, val_dataset)

import yaml
from torch.utils.data import DataLoader
from src.data_pipeline.loader import TrajectoryLoader
from src.data_pipeline.sequence_generator import SocialSequenceGenerator
from src.data_pipeline.dataset import SocialDataset, social_collate
# from src.models.stgcnn import STGCNN
from src.models.social_lstm import SocialLSTM
from src.training.trainer import Trainer

# 1. Setup
print("Loading configuration...")
with open('configs/data_config.yml', 'r') as f:
    config = yaml.safe_load(f)

print("Loading trajectories...")
loader = TrajectoryLoader(config['data']['raw_path'])
generator = SocialSequenceGenerator(obs_len=8, pred_len=12)

print("Loading data...")
# Train on Univ, Validate on Hotel (Simple example)
train_scenes = [
    ('eth', 'univ')
    # ('eth', 'hotel'), 
    # ('ucy', 'zara1'), 
    # ('ucy', 'zara2'), 
    # ('ucy', 'univ')
]
train_dataset = SocialDataset(train_scenes, loader, generator, config)
train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], 
                          shuffle=True, collate_fn=social_collate)

# 2. Initialize Model & Trainer
# model = STGCNN(obs_len=8, pred_len=12)
model = SocialLSTM(obs_len=8, pred_len=12)
trainer = Trainer(model, config)

# 3. Training Loop
print("🚀 Starting Training...")
for epoch in range(config['training']['epochs']):
    avg_loss = trainer.train_epoch(train_loader)
    # if epoch % 10 == 0:
    print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")

# 4. Save Final
trainer.save_model(f"{config['training']['checkpoint_dir']}/lstm_univ.pth")

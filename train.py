import yaml
from torch.utils.data import DataLoader
from src.data_pipeline.loader import TrajectoryLoader
from src.data_pipeline.sequence_generator import SocialSequenceGenerator
from src.data_pipeline.dataset import SocialDataset, social_collate
# from src.models.social_lstm import SocialLSTM
from src.models.stgcnn import STGCNN
from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator

# 1. Setup
with open('configs/data_config.yml', 'r') as f:
    config = yaml.safe_load(f)

loader = TrajectoryLoader(config['data']['raw_path'])
generator = SocialSequenceGenerator(obs_len=8, pred_len=12)

# Leave-One-Out Setup: Train on everything EXCEPT Hotel
# This is how the ETH/UCY benchmark is traditionally performed
train_scenes = [('eth', 'univ'), ('eth', 'hotel'), ('ucy', 'zara2'), ('ucy', 'univ')]
val_scenes = [('ucy', 'zara1')] 

train_dataset = SocialDataset(train_scenes, loader, generator, config)
val_dataset = SocialDataset(val_scenes, loader, generator, config)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")

if len(val_dataset) == 0:
    raise ValueError("Validation dataset is empty! Check scene name or filtering logic.")

train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], 
                          shuffle=True, collate_fn=social_collate)
val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=social_collate)

# 2. Initialize Model & Trainer
# hidden_dim=64 is a good baseline; noise_dim=16 for the K samples
model = STGCNN(obs_len=8, pred_len=12)
trainer = Trainer(model, config)
evaluator = Evaluator(model, config)

# 3. Training Loop
print(f"🚀 Training on {len(train_dataset)} windows...")
for epoch in range(config['training']['epochs']):
    avg_loss = trainer.train_epoch(train_loader)
    
    # Validation every few epochs
    if (epoch + 1) % 5 == 0:
        metrics = evaluator.evaluate(val_loader)
        print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.4f} | ADE: {metrics['ADE']:.3f} | FDE: {metrics['FDE']:.3f}")
        
        # Save best model based on ADE
        if metrics['ADE'] < trainer.best_ade:
            trainer.best_ade = metrics['ADE']
            trainer.save_model(f"{config['training']['checkpoint_dir']}/stgcnn_best_zara1.pth")

print("✅ Training complete.")

import yaml
import torch
from torch.utils.data import DataLoader
from src.data_pipeline.loader import TrajectoryLoader
from src.data_pipeline.sequence_generator import SocialSequenceGenerator
from src.data_pipeline.dataset import SocialDataset, social_collate
from src.models.star import STAR
from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator

# 1. Setup
with open('configs/data_config.yml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize the data components exactly like your train.py
loader = TrajectoryLoader(config['data']['raw_path'])
generator = SocialSequenceGenerator(obs_len=8, pred_len=12)

# Leave-One-Out Setup: Training on all except Zara1 (to match your train.py)
train_scenes = [('eth', 'univ'), ('eth', 'hotel'), ('ucy', 'zara2'), ('ucy', 'univ')]
val_scenes = [('ucy', 'zara1')] 

train_dataset = SocialDataset(train_scenes, loader, generator, config)
val_dataset = SocialDataset(val_scenes, loader, generator, config)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], 
                          shuffle=True, collate_fn=social_collate)
val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=social_collate)

# 2. Initialize STAR Model, Trainer, and Evaluator
# STAR hyperparams: d_model=128 for a good capacity/speed balance on Kaggle
model = STAR(obs_len=8, pred_len=12, d_model=128, nhead=8)
trainer = Trainer(model, config)
evaluator = Evaluator(model, config)

# 3. Training Loop
print(f"🚀 Training STAR on {len(train_dataset)} windows...")
for epoch in range(config['training']['epochs']):
    # trainer.train_epoch handles the loss calculation and optimizer.step()
    avg_loss = trainer.train_epoch(train_loader)
    
    # Validation every 5 epochs
    if (epoch + 1) % 5 == 0:
        metrics = evaluator.evaluate(val_loader)
        print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.4f} | ADE: {metrics['ADE']:.3f} | FDE: {metrics['FDE']:.3f}")
        
        # Save best model based on ADE
        if metrics['ADE'] < trainer.best_ade:
            trainer.best_ade = metrics['ADE']
            checkpoint_path = f"{config['training']['checkpoint_dir']}/star_best_zara1.pth"
            trainer.save_model(checkpoint_path)

print("✅ STAR Training complete.")

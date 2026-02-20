import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from src.data_pipeline.loader import TrajectoryLoader
from src.data_pipeline.sequence_generator import SocialSequenceGenerator
from src.data_pipeline.dataset import SocialDataset, social_collate
from src.models.social_gan import SocialGANGenerator, SocialGANDiscriminator
from src.evaluation.evaluator import Evaluator

# 1. Configuration & Data Setup
with open('configs/data_config.yml', 'r') as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = TrajectoryLoader(config['data']['raw_path'])
generator = SocialSequenceGenerator(obs_len=8, pred_len=12)
train_scenes = [('eth', 'univ'), ('eth', 'hotel'), ('ucy', 'zara2'), ('ucy', 'univ')]
train_dataset = SocialDataset(train_scenes, loader, generator, config)
train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], 
                          shuffle=True, collate_fn=social_collate)

val_scenes = [('ucy', 'zara1')]
val_dataset = SocialDataset(val_scenes, loader, generator, config)
val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=social_collate)

# 2. Model Initialization
# We use the same dimensions we discussed for the thesis chapter
netG = SocialGANGenerator(obs_len=8, pred_len=12).to(device)
netD = SocialGANDiscriminator(obs_len=8, pred_len=12).to(device)

evaluator = Evaluator(netG, config)

# 3. Optimizers & Loss Functions
optimizerG = torch.optim.Adam(netG.parameters(), lr=1e-3)
optimizerD = torch.optim.Adam(netD.parameters(), lr=1e-3)
gan_criterion = nn.BCELoss() # Binary Cross Entropy for Real vs Fake
l2_criterion = nn.MSELoss()  # For the Variety Loss

print(f"🚀 Starting Social-GAN Training on {device}...")

# 4. Training Loop
for epoch in range(config['training']['epochs']):
    netG.train()
    netD.train()
    epoch_loss_g, epoch_loss_d = 0, 0
    total_scenes = 0 # <--- Track total scenes for correct averaging
    
    for i, batch in enumerate(train_loader):
        obs_rel = [r.to(device) for r in batch['obs_rel']]
        target_rel = [p.to(device) for p in batch['pred_rel']]
        
        for obs_r, target_r in zip(obs_rel, target_rel):
            total_scenes += 1 # Increment for every scene
            N = obs_r.size(0)
            
            # --- STEP 1: Train Discriminator ---
            optimizerD.zero_grad()
            real_traj = torch.cat([obs_r, target_r], dim=1)
            real_labels = torch.ones(N, 1).to(device)
            
            output_real = netD(real_traj)
            loss_d_real = gan_criterion(output_real, real_labels)
            
            fake_pred = netG([None], [obs_r], k=1)[0][0]
            fake_traj = torch.cat([obs_r, fake_pred.detach()], dim=1)
            fake_labels = torch.zeros(N, 1).to(device)
            
            output_fake = netD(fake_traj)
            loss_d_fake = gan_criterion(output_fake, fake_labels)
            
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optimizerD.step()
            epoch_loss_d += loss_d.item()

            # --- STEP 2: Train Generator ---
            optimizerG.zero_grad()
            k_samples = netG([None], [obs_r], k=20)[0] 
            
            # Variety Loss (Best-of-K) calculation
            diff = k_samples - target_r.unsqueeze(0)
            dist = torch.norm(diff, p=2, dim=-1)
            ade_k = dist.mean(dim=(1, 2))
            best_sample_idx = torch.argmin(ade_k)
            
            loss_l2 = l2_criterion(k_samples[best_sample_idx], target_r)
            
            # Adversarial Loss
            fake_traj_for_g = torch.cat([obs_r, k_samples[best_sample_idx]], dim=1)
            output_g = netD(fake_traj_for_g)
            loss_adv = gan_criterion(output_g, real_labels)
            
            loss_g = loss_l2 + 0.1 * loss_adv
            loss_g.backward()
            optimizerG.step()
            epoch_loss_g += loss_g.item()

    # --- Evaluation Phase (Scoreboard) ---
    # We switch to .eval() to turn off dropout/batchnorm for consistent metrics
    netG.eval() 
    with torch.no_grad():
        metrics = evaluator.evaluate(val_loader)

    # Fixed: Divide by total_scenes, not len(train_loader)
    avg_g = epoch_loss_g / total_scenes
    avg_d = epoch_loss_d / total_scenes
    print(f"Epoch [{epoch+1}/{50}] | Loss G: {avg_g:.4f} | Loss D: {avg_d:.4f} |"
          f"ADE: {metrics['ADE']:.3f} | FDE: {metrics['FDE']:.3f}")


# 5. Save the trained Generator
torch.save({'state_dict': netG.state_dict()}, 'results/checkpoints/sgan_generator_zara1.pth')
print("✅ Social-GAN Training Complete.")

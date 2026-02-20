import yaml
import torch
import pandas as pd
from torch.utils.data import DataLoader
from src.data_pipeline.loader import TrajectoryLoader
from src.data_pipeline.sequence_generator import SocialSequenceGenerator
from src.data_pipeline.dataset import SocialDataset, social_collate
from src.models.social_lstm import SocialLSTM
from src.models.stgcnn import STGCNN
from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator

# 1. Setup & Config
with open('configs/data_config.yml', 'r') as f:
    config = yaml.safe_load(f)

scenes = [
    ('eth', 'univ'), 
    ('eth', 'hotel'), 
    ('ucy', 'univ'), 
    ('ucy', 'zara1'), 
    ('ucy', 'zara2')
]

loader = TrajectoryLoader(config['data']['raw_path'])
generator = SocialSequenceGenerator(obs_len=8, pred_len=12)

# Choose your model: "LSTM" or "STGCNN"
MODEL_TYPE = "STGCNN" 

results = []

# 2. Leave-One-Out Loop
for i, test_scene in enumerate(scenes):
    train_scenes = [s for j, s in enumerate(scenes) if i != j]
    print(f"\n🚀 Benchmarking {MODEL_TYPE} | Test Scene: {test_scene[1].upper()}")
    print(f"Training on: {[s[1] for s in train_scenes]}")

    # Datasets
    train_dataset = SocialDataset(train_scenes, loader, generator, config)
    test_dataset = SocialDataset([test_scene], loader, generator, config)
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], 
                              shuffle=True, collate_fn=social_collate)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=social_collate)

    # Initialize Model
    if MODEL_TYPE == "LSTM":
        model = SocialLSTM(obs_len=8, pred_len=12, hidden_dim=64)
    else:
        model = STGCNN(obs_len=8, pred_len=12)

    trainer = Trainer(model, config)
    evaluator = Evaluator(model, config)

    # Train
    for epoch in range(config['training']['epochs']):
        _ = trainer.train_epoch(train_loader)
    
    # Final Evaluation
    metrics = evaluator.evaluate(test_loader)
    print(f"✅ Result for {test_scene[1]}: ADE: {metrics['ADE']:.3f}, FDE: {metrics['FDE']:.3f}")
    
    results.append({
        'Scene': test_scene[1],
        'ADE': round(metrics['ADE'], 3),
        'FDE': round(metrics['FDE'], 3)
    })

# 3. Final Table Generation
df = pd.DataFrame(results)
avg_ade = df['ADE'].mean()
avg_fde = df['FDE'].mean()
df.loc[len(df)] = {'Scene': 'AVERAGE', 'ADE': avg_ade, 'FDE': avg_fde}

print("\n" + "="*30)
print(f"FINAL {MODEL_TYPE} BENCHMARK RESULTS")
print("="*30)
print(df.to_string(index=False))
df.to_csv(f"results/{MODEL_TYPE.lower()}_benchmark.csv", index=False)

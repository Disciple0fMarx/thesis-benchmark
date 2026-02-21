import yaml
import torch
import pandas as pd
from torch.utils.data import DataLoader

from src.data_pipeline.loader import TrajectoryLoader
from src.data_pipeline.sequence_generator import SocialSequenceGenerator
from src.data_pipeline.dataset import SocialDataset, social_collate

from src.models.social_lstm import SocialLSTM
from src.models.stgcnn import STGCNN
from src.models.social_gan import SocialGANGenerator
from src.models.star import STAR

from src.evaluation.evaluator import Evaluator


# -----------------------
# 1. Setup
# -----------------------
with open('configs/data_config.yml', 'r') as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scenes = [
    ('ucy', 'zara1')   # since weights are for zara1
]

# 🔥 Explicit checkpoint mapping
CHECKPOINTS = {
    "LSTM": "results/checkpoints/lstm_best_zara1.pth",
    "STGCNN": "results/checkpoints/stgcnn_best_zara1.pth",
    "SocialGAN": "results/checkpoints/sgan_generator_zara1.pth",
    "STAR": "results/checkpoints/star_best_zara1.pth",
}

loader = TrajectoryLoader(config['data']['raw_path'])
generator = SocialSequenceGenerator(obs_len=8, pred_len=12)

all_results = []


# -----------------------
# 2. Model Factory
# -----------------------
def build_model(model_name):
    if model_name == "LSTM":
        return SocialLSTM(obs_len=8, pred_len=12, hidden_dim=64)

    elif model_name == "STGCNN":
        return STGCNN(obs_len=8, pred_len=12)

    elif model_name == "SocialGAN":
        return SocialGANGenerator(obs_len=8, pred_len=12)

    elif model_name == "STAR":
        return STAR(obs_len=8, pred_len=12)

    else:
        raise ValueError(f"Unknown model: {model_name}")


# -----------------------
# 3. Benchmark Loop
# -----------------------
for model_name, checkpoint_path in CHECKPOINTS.items():

    print("\n" + "=" * 50)
    print(f"🚀 Evaluating Model: {model_name}")
    print("=" * 50)

    model_results = []

    for test_scene in scenes:

        print(f"  ➤ Scene: {test_scene[1].upper()}")

        test_dataset = SocialDataset([test_scene], loader, generator, config)
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            collate_fn=social_collate
        )

        model = build_model(model_name)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Your checkpoints store weights under "state_dict"
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()

        evaluator = Evaluator(model, config)

        with torch.no_grad():
            metrics = evaluator.evaluate(test_loader)

        print(f"     ADE: {metrics['ADE']:.3f} | FDE: {metrics['FDE']:.3f}")

        model_results.append({
            'Model': model_name,
            'Scene': test_scene[1],
            'ADE': metrics['ADE'],
            'FDE': metrics['FDE']
        })

    all_results.extend(model_results)


# -----------------------
# 4. Final Table
# -----------------------
df = pd.DataFrame(all_results)

print("\n" + "=" * 60)
print("🏆 FINAL BENCHMARK RESULTS")
print("=" * 60)
print(df.to_string(index=False))

df.to_csv("results/zara1_model_comparison.csv", index=False)

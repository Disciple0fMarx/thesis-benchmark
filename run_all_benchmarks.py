import torch
import pandas as pd
import yaml
from src.models import ConstantVelocityModel, STGCNN, SocialLSTM
from src.evaluation.evaluator import Evaluator


def run_benchmarks(test_scene=('eth', 'univ')):
    with open('configs/data_config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Configuration
    models_to_test = {
        "CVM": ConstantVelocityModel(pred_len=12),
        "STGCNN": STGCNN(obs_len=8, pred_len=12),
        "Social-LSTM": SocialLSTM(obs_len=8, pred_len=12)
    }
    
    # Load trained weights for DL models
    # (Assuming you've saved them as stgcnn_final.pth and lstm_final.pth)
    try:
        models_to_test["STGCNN"].load_state_dict(torch.load('results/checkpoints/stgcnn_hotel.pth'))
        models_to_test["Social-LSTM"].load_state_dict(torch.load('results/checkpoints/lstm_hotel.pth'))
    except:
        print("⚠️ DL weights not found, running with random initialization for demo.")

    results_table = []

    for name, model in models_to_test.items():
        evaluator = Evaluator(model, config)
        metrics = evaluator.evaluate(test_loader)
        results_table.append({
            "Model": name,
            "ADE": round(metrics['ADE'], 4),
            "FDE": round(metrics['FDE'], 4)
        })

    df = pd.DataFrame(results_table)
    print("\n📝 FINAL BENCHMARK TABLE")
    print(df.to_markdown(index=False))
    return df

# Run it
run_benchmarks()

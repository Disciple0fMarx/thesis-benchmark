from src.data_pipeline.loader import TrajectoryLoader
from src.data_pipeline.sequence_generator import SocialSequenceGenerator
import yaml

# 1. Load Config
with open('configs/data_config.yml', 'r') as f:
    config = yaml.safe_load(f)

# 2. Extract Data using YOUR loader
loader = TrajectoryLoader(config['data']['raw_path'])
raw_df = loader.load_scene('eth', 'univ')

# 3. Pass to Generator
gen = SocialSequenceGenerator(
    obs_len=config['data']['t_obs'], 
    pred_len=config['data']['t_pred'],
    step=config['data']['sliding_step']
)
sequences = gen.generate(raw_df)

print(f"Loaded {len(sequences)} windows from ETH Univ.")
print(f"Each window contains [Peds, {config['data']['t_obs'] + config['data']['t_pred']}, 2] coordinates.")

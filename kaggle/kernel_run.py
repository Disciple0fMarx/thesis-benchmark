# kaggle/kernel.py
import subprocess, os, sys

# Pull the repo
subprocess.run([
    "git", "clone",
    "https://github.com/Disciple0fMarx/thesis-benchmark.git",
    "/kaggle/working/repo"
], check=True)

os.chdir("/kaggle/working/repo")
sys.path.insert(0, "/kaggle/working/repo")

# Install dependencies
subprocess.run([
    "pip", "install", "einops", "pyyaml", "--quiet"
], check=True)

# Copy the dataset (uploaded once to Kaggle datasets)
subprocess.run([
    "cp", "-r",
    "/kaggle/input/eth-ucy-trajectories/",
    "data/raw"
], check=True)

# Run training for one subset
subset = "univ"
subprocess.run([
    "python", "train_moflow_teacher.py",
    "--subset", subset,
    "--override", "training.batch_size=128",
    "--override", "evaluation.ode_steps=10",
], check=True)

# Output lives in /kaggle/working/repo/results/
# Kaggle automatically saves everything in /kaggle/working/ as output
import shutil
shutil.copytree(
    "results/moflow",
    "/kaggle/working/results",
    dirs_exist_ok=True
)


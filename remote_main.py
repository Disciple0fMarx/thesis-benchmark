import os
import subprocess

# 1. Unzip the code/data provided by the dataset
os.system("unzip /kaggle/input/thesis-code-zip/thesis_v1.zip -d /kaggle/working/")

# 2. Create the output directory structure so the script doesn't crash
os.makedirs("/kaggle/working/results/checkpoints", exist_ok=True)
os.makedirs("/kaggle/working/results/plots", exist_ok=True)

# 3. Install requirements
os.system("pip install -r /kaggle/working/requirements.txt")

# 4. Run the training
# We use -u to get real-time logs in the Kaggle console
process = subprocess.Popen(
    ["python", "-u", "/kaggle/working/train_gan.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

for line in process.stdout:
    print(line, end="")

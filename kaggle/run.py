# kaggle/run.py
import json, time, argparse
import kaggle

api = kaggle.api
api.authenticate()

USERNAME = "dhyaelbahri"


def push_and_run(subset: str):
    kernel_slug = f"moflow-teacher-{subset}"

    kernel_meta = {
        "id": f"{USERNAME}/{kernel_slug}",
        "title": f"MoFlow Teacher {subset}",
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [f"{USERNAME}/eth-ucy-trajectories"],
        "kernel_sources": [],
        "competition_sources": [],
        "total_votes": 0,
    }

    with open("kaggle/kernel-metadata.json", "w") as f:
        json.dump(kernel_meta, f, indent=2)

    # Inject the subset as an env var via the kernel script
    with open("kaggle/kernel.py", "r") as f:
        src = f.read()
    patched = src.replace(
        'os.environ.get("SUBSET", "eth")',
        f'"{subset}"'
    )
    with open("kaggle/kernel_run.py", "w") as f:
        f.write(patched)

    kernel_meta["code_file"] = "kernel_run.py"
    with open("kaggle/kernel-metadata.json", "w") as f:
        json.dump(kernel_meta, f, indent=2)

    print(f"Pushing kernel for subset={subset}...")
    api.kernels_push("kaggle")

    # Poll until complete
    print("Waiting for kernel to finish (this will take a while)...")
    while True:
        status = api.kernels_status(f"{USERNAME}/{kernel_slug}")
        s = status.status
        print(f"  Status: {s}")
        if s in ("complete", "error", "cancel"):
            break
        time.sleep(60)

    if s == "complete":
        print("Downloading output...")
        api.kernels_output(
            f"{USERNAME}/{kernel_slug}",
            path=f"results/kaggle/{subset}"
        )
        print(f"Weights saved to results/kaggle/{subset}/")
    else:
        print(f"Kernel failed with status: {s}")
        print("Check https://www.kaggle.com/code for logs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", required=True)
    args = parser.parse_args()
    push_and_run(args.subset)


import os
from matminer.datasets import load_dataset
import joblib

def download_matbench_mp_gap(save_path="data/matbench_mp_gap.pkl"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print("Downloading Matbench 'matbench_mp_gap' dataset (~106k entries)...")
    df = load_dataset("matbench_mp_gap")
    print(f"Downloaded {len(df)} records. Saving to {save_path}...")
    joblib.dump(df, save_path)
    print("Dataset saved.")

if __name__ == "__main__":
    download_matbench_mp_gap()

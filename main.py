import argparse
from pathlib import Path
from src.train import run_training

def parse_args():
    p = argparse.ArgumentParser(description="Train GNN for MatBench band gap prediction")
    p.add_argument("--property", type=str, default="band_gap", choices=["band_gap"], help="Target property")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--radius", type=float, default=8.0, help="Neighbor cutoff in Å")
    p.add_argument("--max-neighbors", type=int, default=12)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=3, help="Number of GNN layers")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--save-dir", type=str, default="models")
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--test-split", type=float, default=0.1)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    run_training(args)

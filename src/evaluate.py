from __future__ import annotations
from pathlib import Path
import argparse
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from torch_geometric.loader import DataLoader
from .dataloader import create_dataloaders, GraphConfig
from .model import GNNRegressor

def load_model(ckpt_path: str, device: str = "cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = GNNRegressor(node_dim=ckpt["node_dim"], edge_dim=ckpt["edge_dim"], hidden=ckpt["hidden"], layers=ckpt["layers"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    _, _, test_loader = create_dataloaders(root="data", batch_size=args.batch_size, val_split=0.1, test_split=0.1, seed=42, config=GraphConfig())
    model = load_model(args.ckpt, device=device)

    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch).detach().cpu()
            all_pred.append(pred)
            all_true.append(batch.y.view(-1).detach().cpu())
    import numpy as np
    y_pred = torch.cat(all_pred).numpy()
    y_true = torch.cat(all_true).numpy()

    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"TEST: MAE={mae:.4f} R2={r2:.4f}")

    # Parity plot
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.4)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, linestyle="--")
    plt.xlabel("True band gap (eV)")
    plt.ylabel("Predicted band gap (eV)")
    plt.title("Parity Plot: Band Gap")
    plt.savefig("models/parity_plot.png", dpi=160)
    print("Saved parity plot to models/parity_plot.png")

if __name__ == "__main__":
    main()

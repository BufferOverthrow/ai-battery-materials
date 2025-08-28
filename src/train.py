from __future__ import annotations
from pathlib import Path
import torch
from torch import nn, optim
from torch_geometric.data import Data
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error
from .dataloader import create_dataloaders, GraphConfig
from .model import GNNRegressor

def _device(dev_str: str) -> torch.device:
    return torch.device(dev_str if torch.cuda.is_available() and dev_str == "cuda" else "cpu")

def _step(model, batch, criterion, device):
    batch = batch.to(device)
    pred = model(batch)
    loss = criterion(pred, batch.y.view(-1))
    return loss, pred.detach().cpu(), batch.y.view(-1).detach().cpu()

def run_training(args):
    device = _device(args.device)
    cfg = GraphConfig(radius=args.radius, max_neighbors=args.max_neighbors)

    train_loader, val_loader, test_loader = create_dataloaders(root="data", batch_size=args.batch_size, val_split=args.val_split, test_split=args.test_split, seed=args.seed, config=cfg)

    # Inspect node/edge dims from a single batch
    sample = next(iter(train_loader))
    node_dim = sample.x.shape[-1]
    edge_dim = sample.edge_attr.shape[-1]

    model = GNNRegressor(node_dim=node_dim, edge_dim=edge_dim, hidden=args.hidden, layers=args.layers).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val = float("inf")
    save_path = Path(args.save_dir) / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]"):
            optimizer.zero_grad()
            loss, _, _ = _step(model, batch, criterion, device)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        all_pred, all_true = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]"):
                loss, pred, true = _step(model, batch, criterion, device)
                val_losses.append(loss.item())
                all_pred.append(pred)
                all_true.append(true)
        all_pred = torch.cat(all_pred).numpy()
        all_true = torch.cat(all_true).numpy()
        val_mae = mean_absolute_error(all_true, all_pred)
        val_r2 = r2_score(all_true, all_pred)
        scheduler.step(val_mae)

        print(f"Epoch {epoch}: train_loss={sum(train_losses)/len(train_losses):.4f} val_MAE={val_mae:.4f} val_R2={val_r2:.4f}")

        if val_mae < best_val:
            best_val = val_mae
            torch.save({"model_state": model.state_dict(),
                        "node_dim": node_dim,
                        "edge_dim": edge_dim,
                        "hidden": args.hidden,
                        "layers": args.layers}, save_path)
            print(f"Saved new best checkpoint to {save_path}")

    # Final test
    model.load_state_dict(torch.load(save_path)["model_state"])
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test"):
            _, pred, true = _step(model, batch, criterion, device)
            all_pred.append(pred)
            all_true.append(true)
    all_pred = torch.cat(all_pred).numpy()
    all_true = torch.cat(all_true).numpy()
    test_mae = mean_absolute_error(all_true, all_pred)
    test_r2 = r2_score(all_true, all_pred)
    print(f"TEST: MAE={test_mae:.4f} R2={test_r2:.4f}")

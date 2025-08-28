# AI for Next-Gen Battery Materials

**Date**: 2025-08-28

A modular pipeline to **predict** battery-material properties (starting with **band gap** from MatBench)
using **Graph Neural Networks (PyTorch Geometric)** — with room to expand into **generative design** and **simulation**.

---

## Quickstart

### 1) Create an environment & install deps
```bash
python -m venv .venv && source .venv/bin/activate  # (Linux/Mac)
# or: py -m venv .venv && .venv\Scripts\activate  # (Windows)

pip install -r requirements.txt
# For torch-geometric, follow the wheel instructions printed at install time if needed.
```

### 2) Train band-gap predictor
```bash
python main.py --property band_gap --epochs 50 --batch-size 64
```

### 3) Evaluate
```bash
python -m src.evaluate --ckpt models/best.pt
```

---

## Project Structure
```
battery-ai/
├── data/                # Raw and processed data
├── models/              # Saved models
├── notebooks/           # EDA, experiments
├── src/
│   ├── __init__.py
│   ├── dataloader.py    # Dataset utils
│   ├── model.py         # GNN architecture
│   ├── train.py         # Training loop
│   └── evaluate.py      # Metrics and plots
├── main.py              # Script to run training
├── requirements.txt
└── README.md
```

---

## Notes

- We start with **MatBench band gap** (`matbench_v0.1_band_gap`).
- Graph construction uses **radius neighbors** (8 Å, max 12) with **Gaussian distance basis** as edge features.
- Model is a **PyG NNConv stack** + global pooling + MLP head (regression).
- Loss: **MAE**. Metrics: MAE/R².
- If you have CUDA, install the Torch + PyG wheels that match your CUDA version for best performance.

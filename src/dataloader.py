from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List
import math
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
import numpy as np

# External libs expected at runtime
# - matminer, pymatgen
from matminer.datasets import load_dataset
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN

def _gaussian_rbf(distances: torch.Tensor, centers: torch.Tensor, gamma: float = 10.0) -> torch.Tensor:
    # distances: [E], centers: [K]
    d = distances.view(-1, 1) - centers.view(1, -1)
    return torch.exp(-gamma * (d ** 2))

def _atom_feature_Z(z: int, max_z: int = 100) -> torch.Tensor:
    # Simple one-hot for atomic number up to max_z
    feat = torch.zeros(max_z, dtype=torch.float32)
    if 1 <= z <= max_z:
        feat[z - 1] = 1.0
    return feat

@dataclass
class GraphConfig:
    radius: float = 8.0
    max_neighbors: int = 12
    rbf_k: int = 32
    rbf_gamma: float = 10.0

class MatBenchBandGapDataset(InMemoryDataset):
    def __init__(self, root: str, transform=None, pre_transform=None, config: GraphConfig | None = None):
        self.config = config or GraphConfig()
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ["dummy"]  # managed by matminer on-the-fly

    @property
    def processed_file_names(self) -> List[str]:
        return ["matbench_bandgap.pt"]

    def download(self):
        # Matminer loads dataset directly; nothing to download here.
        return

    def process(self):
        df = load_dataset("matbench_v0.1_band_gap")
        # df columns: 'structure' (pymatgen Structure), 'band_gap'
        data_list: List[Data] = []
        rbf_centers = torch.linspace(0, self.config.radius, self.config.rbf_k)

        for idx, row in df.iterrows():
            struct: Structure = row["structure"]
            y = float(row["band_gap"])

            # Build graph
            positions = []
            atom_feats = []
            for site in struct.sites:
                positions.append(site.frac_coords)  # fractional
                atom_feats.append(_atom_feature_Z(site.specie.Z))

            positions = torch.tensor(np.array(struct.frac_coords), dtype=torch.float32)
            # Convert to cart coords for distances
            cart_coords = torch.tensor(np.array([s.coords for s in struct.sites]), dtype=torch.float32)
            x = torch.stack(atom_feats, dim=0)  # [N, max_z]

            # Neighbor list via simple radius in cartesian space
            # Build edges
            edge_index_src = []
            edge_index_dst = []
            edge_attr_dist = []

            N = cart_coords.shape[0]
            for i in range(N):
                # brute-force neighbor search (can be optimized)
                dvec = cart_coords - cart_coords[i]
                dists = torch.linalg.norm(dvec, dim=1)
                # sort neighbors by distance, exclude self (dist=0)
                order = torch.argsort(dists)
                cnt = 0
                for j in order:
                    j = int(j)
                    if i == j:
                        continue
                    if dists[j].item() <= self.config.radius:
                        edge_index_src.append(i)
                        edge_index_dst.append(j)
                        edge_attr_dist.append(dists[j].item())
                        cnt += 1
                        if cnt >= self.config.max_neighbors:
                            break

            if len(edge_index_src) == 0:
                # fallback to at least a self-loop to avoid empty graphs
                edge_index_src = [0]
                edge_index_dst = [0]
                edge_attr_dist = [0.0]

            edge_index = torch.tensor([edge_index_src, edge_index_dst], dtype=torch.long)
            distances = torch.tensor(edge_attr_dist, dtype=torch.float32)
            edge_attr = _gaussian_rbf(distances, rbf_centers, gamma=self.config.rbf_gamma)  # [E, K]

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([y], dtype=torch.float32))
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def create_dataloaders(root: str, batch_size: int = 64, val_split: float = 0.1, test_split: float = 0.1, seed: int = 42, config: GraphConfig | None = None):
    dataset = MatBenchBandGapDataset(root=root, config=config)
    n = len(dataset)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_val - n_test

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [n_train, n_val, n_test], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

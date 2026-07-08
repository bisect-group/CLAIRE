import random
from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class LayerNormNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int, num_layers: int, drop_out: float = 0.1):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")
        layers = [
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )
        ]
        for _ in range(num_layers - 2):
            layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                )
            )
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = self.dropout(layer(x))
        return self.layers[-1](x)


class TripletFeatureDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: Sequence[str]):
        if len(features) != len(labels):
            raise ValueError("features and labels must have equal length")
        self.features = features.astype(np.float32, copy=False)
        self.labels = list(labels)
        self.by_label: Dict[str, List[int]] = defaultdict(list)
        for index, label in enumerate(self.labels):
            self.by_label[label].append(index)
        self.label_values = sorted(self.by_label)
        if len(self.label_values) < 2:
            raise ValueError("Triplet training requires at least two labels")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        label = self.labels[index]
        positive_index = random.choice(self.by_label[label])
        negative_label = random.choice(self.label_values)
        while negative_label == label:
            negative_label = random.choice(self.label_values)
        negative_index = random.choice(self.by_label[negative_label])
        return (
            torch.from_numpy(self.features[index]),
            torch.from_numpy(self.features[positive_index]),
            torch.from_numpy(self.features[negative_index]),
        )


@torch.no_grad()
def embed_features(model: nn.Module, features: np.ndarray, device: torch.device, batch_size: int = 8192) -> torch.Tensor:
    model.eval()
    chunks = []
    for start in range(0, len(features), batch_size):
        batch = torch.from_numpy(features[start : start + batch_size]).to(device=device, dtype=torch.float32)
        chunks.append(model(batch).detach().cpu())
    return torch.cat(chunks, dim=0)

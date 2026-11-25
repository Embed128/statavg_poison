"""Official StatAvg 3-hidden-layer MLP architecture in PyTorch.

Matches Bouzinis et al. (IEEE TNSM 2025) and Flower baselines/statavg/model.py:
  Linear(D, 128) -> ReLU -> Linear(128, 128) -> ReLU -> Linear(128, 128) -> ReLU -> Linear(128, C)
  Optimizer: Adam(lr=0.002, betas=(0.99, 0.999), eps=1e-8)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StatAvgMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

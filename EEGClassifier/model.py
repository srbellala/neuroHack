"""
MLP emotion classifier.

Input  : 16-dimensional DE feature vector
         (4 channels × 4 bands: theta, alpha, beta, gamma)
Output : 3-class logits  →  calm(0) / focused(1) / stressed(2)
"""

import torch
import torch.nn as nn

# Maps integer class index → EmotionalState string value
IDX_TO_STATE: dict[int, str] = {0: "calm", 1: "focused", 2: "stressed"}
STATE_TO_IDX: dict[str, int] = {v: k for k, v in IDX_TO_STATE.items()}


class EmotionMLP(nn.Module):
    """Feedforward MLP with BatchNorm and Dropout.

    Architecture
    ------------
    16 → 128 → 64 → 32 → 3
    Each hidden layer: Linear → BatchNorm1d → ReLU → Dropout
    """

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dims: tuple[int, ...] = (128, 64, 32),
        num_classes: int = 3,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, input_dim)

        Returns
        -------
        logits : (batch, num_classes)
        """
        return self.net(x)

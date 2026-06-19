"""Occupancy and semantic prediction heads for Phase B.

These heads are added to ReSplat's Gaussian adapter to predict:
- Per-Gaussian occupancy probability (replacing opacity)
- Per-Gaussian semantic class logits (replacing SH color)

The heads take Gaussian parameters as input and output predictions.
They are trained end-to-end with the VGGT geometric supervision.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class OccupancyHead(nn.Module):
    """Predicts per-Gaussian occupancy probability.

    Replaces the opacity parameter from the original Gaussian adapter.
    Instead of alpha-blending weight (entangled with color), this predicts
    a proper occupancy probability via the probabilistic field formulation.

    Input features:
        - Gaussian means (3)
        - Gaussian scales (3)
        - Gaussian rotations (4, quaternion)
        - Optional: identity encoding (D_id)

    Output:
        - Occupancy probability in [0, 1] via sigmoid

    Phase A.1 POC showed that post-hoc MLP on frozen Gaussians fails (2.6%
    near surface). This head must be trained end-to-end with Gaussian
    position updates for meaningful occupancy prediction.
    """

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ):
        """Initialize occupancy head.

        Args:
            input_dim: Input feature dimension. Default 10 (3 means + 3 scales + 4 rotation).
            hidden_dims: Hidden layer dimensions. Default [64, 32].
            dropout: Dropout rate for regularization.
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        # Final layer: single logit (sigmoid applied externally for flexibility)
        layers.append(nn.Linear(in_dim, 1))

        self.mlp = nn.Sequential(*layers)

        # Initialize final layer with small weights for stable early training
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, -2.0)  # sigmoid(-2) ≈ 0.12, biased toward low occupancy

    def forward(
        self,
        means: Tensor,
        scales: Tensor,
        rotations: Tensor,
        extra_features: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict occupancy probability for each Gaussian.

        Args:
            means: (N, 3) Gaussian centers.
            scales: (N, 3) Gaussian scales.
            rotations: (N, 4) Gaussian rotations (quaternion).
            extra_features: (N, D) optional additional features.

        Returns:
            (N,) occupancy probabilities in [0, 1].
        """
        # Concatenate Gaussian parameters
        features = torch.cat([means, scales, rotations], dim=-1)  # (N, 10)

        if extra_features is not None:
            features = torch.cat([features, extra_features], dim=-1)

        # MLP predicts logit
        logit = self.mlp(features).squeeze(-1)  # (N,)

        # Sigmoid to get probability
        return torch.sigmoid(logit)


class SemanticHead(nn.Module):
    """Predicts per-Gaussian semantic class logits.

    Replaces the spherical harmonics (color) from the original Gaussian adapter.
    Instead of photorealistic color, this predicts semantic class distribution
    for planning-oriented costmap generation.

    Input features:
        - Gaussian means (3)
        - Gaussian scales (3)
        - Gaussian rotations (4, quaternion)
        - Optional: identity encoding (D_id)

    Output:
        - K-class logits (before softmax)

    Classes follow COCO taxonomy (mapped from SAM2/YOLO labels).
    """

    def __init__(
        self,
        num_classes: int,
        input_dim: int = 10,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ):
        """Initialize semantic head.

        Args:
            num_classes: Number of semantic classes (K).
            input_dim: Input feature dimension. Default 10.
            hidden_dims: Hidden layer dimensions. Default [64, 32].
            dropout: Dropout rate.
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.num_classes = num_classes

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        # Final layer: K logits
        layers.append(nn.Linear(in_dim, num_classes))

        self.mlp = nn.Sequential(*layers)

        # Xavier initialization for classification head
        nn.init.xavier_uniform_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        means: Tensor,
        scales: Tensor,
        rotations: Tensor,
        extra_features: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict semantic class logits for each Gaussian.

        Args:
            means: (N, 3) Gaussian centers.
            scales: (N, 3) Gaussian scales.
            rotations: (N, 4) Gaussian rotations (quaternion).
            extra_features: (N, D) optional additional features.

        Returns:
            (N, K) raw logits for K semantic classes.
        """
        features = torch.cat([means, scales, rotations], dim=-1)

        if extra_features is not None:
            features = torch.cat([features, extra_features], dim=-1)

        return self.mlp(features)


class ConfidenceHead(nn.Module):
    """Predicts per-Gaussian confidence/uncertainty.

    Optional head that predicts the confidence of the occupancy prediction.
    Used for:
    - Down-weighting uncertain Gaussians in the loss
    - Identifying regions that need more Gaussians (RL density allocation)
    - Uncertainty-aware BEV projection

    Output: confidence score in [0, 1] (higher = more certain).
    """

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [32, 16]

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

        # Initialize toward high confidence (sigmoid(2) ≈ 0.88)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, 2.0)

    def forward(
        self,
        means: Tensor,
        scales: Tensor,
        rotations: Tensor,
        extra_features: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict confidence for each Gaussian.

        Returns:
            (N,) confidence scores in [0, 1].
        """
        features = torch.cat([means, scales, rotations], dim=-1)

        if extra_features is not None:
            features = torch.cat([features, extra_features], dim=-1)

        logit = self.mlp(features).squeeze(-1)
        return torch.sigmoid(logit)

"""Phase B training module for cross-model geometric distillation.

This module implements the loss functions, occupancy/semantic heads,
and training infrastructure for retraining ReSplat with VGGT-Ω
geometric supervision.

Key components:
    - losses: Chamfer, Focal, Hinge, CE, L1+SSIM loss functions
    - heads: Occupancy and semantic prediction heads
    - supervision: VGGT-based per-Gaussian labeling
    - trainer: Three-stage training loop
"""

from eof3r.src.training.losses import (
    chamfer_depth_loss,
    focal_occupancy_loss,
    hinge_free_space_loss,
    semantic_cross_entropy_loss,
    color_reconstruction_loss,
    compute_total_loss,
)
from eof3r.src.training.heads import OccupancyHead, SemanticHead
from eof3r.src.training.supervision import label_gaussians_by_vggt_projection

__all__ = [
    "chamfer_depth_loss",
    "focal_occupancy_loss",
    "hinge_free_space_loss",
    "semantic_cross_entropy_loss",
    "color_reconstruction_loss",
    "compute_total_loss",
    "OccupancyHead",
    "SemanticHead",
    "label_gaussians_by_vggt_projection",
]

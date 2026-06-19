"""Loss functions for Phase B cross-model geometric distillation.

Implements the probabilistic occupancy field loss derivation from
docs/lit_notes/phaseb_design_2026-06-19.md §2.

Each Gaussian defines an occupancy field:
    p_i(x) = o_i * N(x; mu_i, Sigma_i)

The total loss is:
    L_total = α·L_depth + β·L_occ + γ·L_free + δ·L_sem + η·L_color

All losses operate on torch tensors and are differentiable.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


# ---------------------------------------------------------------------------
# L_depth: Bidirectional Chamfer Distance
# ---------------------------------------------------------------------------

def chamfer_depth_loss(
    gaussian_means: Tensor,
    vggt_points: Tensor,
    occupied_mask: Optional[Tensor] = None,
) -> Tensor:
    """Bidirectional Chamfer Distance between Gaussian means and VGGT surface points.

    From Phase B design §2.4:
        L_depth = (1/|P|) Σ_p min_i ||μ_i - p||² + (1/|O|) Σ_{i∈O} min_p ||μ_i - p||²

    Forward term: each VGGT surface point needs at least one Gaussian nearby.
    Backward term: each occupied Gaussian needs to be near some VGGT surface
    point — penalizes floaters.

    Args:
        gaussian_means: (N, 3) Gaussian center positions in world coords.
        vggt_points: (M, 3) VGGT surface point cloud.
        occupied_mask: (N,) bool mask for occupied Gaussians. If None, all
            Gaussians are used in the backward term.

    Returns:
        Scalar chamfer distance loss.
    """
    if gaussian_means.ndim != 2 or vggt_points.ndim != 2:
        raise ValueError(f"Expected 2D tensors, got {gaussian_means.shape}, {vggt_points.shape}")
    if gaussian_means.shape[1] != 3 or vggt_points.shape[1] != 3:
        raise ValueError(f"Expected 3D points, got {gaussian_means.shape[1]}, {vggt_points.shape[1]}")

    # Forward: for each VGGT point, find nearest Gaussian
    # (M, 1, 3) - (1, N, 3) -> (M, N) -> (M,)
    diff_fwd = vggt_points.unsqueeze(1) - gaussian_means.unsqueeze(0)  # (M, N, 3)
    dist_fwd = (diff_fwd ** 2).sum(dim=-1)  # (M, N)
    min_dist_fwd = dist_fwd.min(dim=1).values  # (M,)
    loss_fwd = min_dist_fwd.mean()

    # Backward: for each Gaussian, find nearest VGGT point
    if occupied_mask is not None:
        means_bwd = gaussian_means[occupied_mask]
    else:
        means_bwd = gaussian_means

    if means_bwd.shape[0] == 0:
        return loss_fwd

    diff_bwd = means_bwd.unsqueeze(1) - vggt_points.unsqueeze(0)  # (|O|, M, 3)
    dist_bwd = (diff_bwd ** 2).sum(dim=-1)  # (|O|, M)
    min_dist_bwd = dist_bwd.min(dim=1).values  # (|O|,)
    loss_bwd = min_dist_bwd.mean()

    return loss_fwd + loss_bwd


# ---------------------------------------------------------------------------
# L_occ: Focal Loss for Occupancy Classification
# ---------------------------------------------------------------------------

def focal_occupancy_loss(
    predicted_occupancy: Tensor,
    labels: Tensor,
    gamma: float = 2.0,
    class_weights: Optional[Tensor] = None,
) -> Tensor:
    """Focal Loss for per-Gaussian occupancy classification.

    From Phase B design §2.4:
        L_occ^focal = -(1/|L|) Σ_i [y_i(1-o_i)^γ log(o_i) + (1-y_i)o_i^γ log(1-o_i)]

    Handles the severe class imbalance (3.6% occupied vs 96.4% free) by
    automatically down-weighting easy-to-classify examples.

    Args:
        predicted_occupancy: (N,) predicted occupancy values in [0, 1].
        labels: (N,) binary labels (1=occupied, 0=free). Must be {0, 1}.
        gamma: Focal loss focusing parameter. Higher = more focus on hard examples.
        class_weights: Optional (2,) tensor [w0, w1] for class balancing.
            If None, uses inverse frequency weighting.

    Returns:
        Scalar focal loss.
    """
    if predicted_occupancy.shape != labels.shape:
        raise ValueError(f"Shape mismatch: {predicted_occupancy.shape} vs {labels.shape}")

    # Clamp to avoid log(0)
    eps = 1e-6
    o = predicted_occupancy.clamp(eps, 1 - eps)
    y = labels.float()

    # Focal modulating factor
    p_t = o * y + (1 - o) * (1 - y)  # probability of true class
    modulating = (1 - p_t) ** gamma

    # Binary cross-entropy per element
    bce = -(y * torch.log(o) + (1 - y) * torch.log(1 - o))

    # Focal loss per element
    focal = modulating * bce

    # Class balancing weights
    if class_weights is not None:
        w = class_weights[0] * (1 - y) + class_weights[1] * y
        focal = focal * w
    else:
        # Inverse frequency weighting
        n_pos = y.sum().clamp(min=1)
        n_neg = (1 - y).sum().clamp(min=1)
        n_total = y.shape[0]
        w1 = n_total / (2 * n_pos)
        w0 = n_total / (2 * n_neg)
        focal = focal * (w0 * (1 - y) + w1 * y)

    return focal.mean()


# ---------------------------------------------------------------------------
# L_free: Squared Hinge Loss for Free-Space Regularization
# ---------------------------------------------------------------------------

def hinge_free_space_loss(
    predicted_occupancy: Tensor,
    free_mask: Tensor,
    epsilon: float = 0.05,
) -> Tensor:
    """Squared hinge loss for free-space Gaussians.

    From Phase B design §2.4:
        L_free = (1/|F|) Σ_{i∈F} max(0, o_i - ε)²

    Free-space Gaussians should have low occupancy, but we don't force them
    to exactly 0 — the hinge threshold ε provides a margin of tolerance.

    Why squared hinge (not BCE)?
    - Free-space labeling comes from "ray not blocked", not "definitely empty"
    - Hinge loss says "below ε is good enough" — more robust to label noise
    - ε = 0.05: close to 0 but not forced to exactly 0

    Args:
        predicted_occupancy: (N,) predicted occupancy values.
        free_mask: (N,) bool mask identifying free-space Gaussians.
        epsilon: Hinge threshold. Occupancy below ε incurs no loss.

    Returns:
        Scalar hinge loss.
    """
    if predicted_occupancy.shape != free_mask.shape:
        raise ValueError(f"Shape mismatch: {predicted_occupancy.shape} vs {free_mask.shape}")

    if not free_mask.any():
        return torch.tensor(0.0, device=predicted_occupancy.device)

    free_occ = predicted_occupancy[free_mask]
    violations = torch.clamp(free_occ - epsilon, min=0.0)
    return (violations ** 2).mean()


# ---------------------------------------------------------------------------
# L_semantic: Per-Gaussian Semantic Cross-Entropy
# ---------------------------------------------------------------------------

def semantic_cross_entropy_loss(
    semantic_logits: Tensor,
    semantic_labels: Tensor,
    occupied_mask: Tensor,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Cross-entropy loss for per-Gaussian semantic classification.

    From Phase B design §2.4:
        L_sem = -(1/|O|) Σ_{i∈O} log softmax(s_i)_{c_i}

    Only applied to occupied Gaussians (free-space has no semantic meaning).

    Args:
        semantic_logits: (N, K) raw logits for K semantic classes.
        semantic_labels: (N,) integer class labels in [0, K).
        occupied_mask: (N,) bool mask for occupied Gaussians.
        label_smoothing: Label smoothing factor (0.0 = no smoothing).

    Returns:
        Scalar cross-entropy loss.
    """
    if not occupied_mask.any():
        return torch.tensor(0.0, device=semantic_logits.device)

    logits_occ = semantic_logits[occupied_mask]  # (|O|, K)
    labels_occ = semantic_labels[occupied_mask]  # (|O|,)

    return F.cross_entropy(
        logits_occ,
        labels_occ,
        label_smoothing=label_smoothing,
        reduction="mean",
    )


# ---------------------------------------------------------------------------
# L_color: Auxiliary Photometric Loss (L1 + SSIM)
# ---------------------------------------------------------------------------

def color_reconstruction_loss(
    rendered: Tensor,
    ground_truth: Tensor,
    ssim_weight: float = 0.2,
) -> Tensor:
    """Auxiliary photometric loss to prevent encoder drift.

    From Phase B design §2.4:
        L_color = λ · (1/HW) Σ [|I_rend - I_gt|_1 + λ_ssim · (1 - SSIM(I_rend, I_gt))]

    This is an auxiliary loss (η=0.1) that keeps the cost volume encoder
    from losing its pretrained geometric knowledge during occupancy training.

    Args:
        rendered: (B, 3, H, W) rendered images in [0, 1].
        ground_truth: (B, 3, H, W) ground truth images in [0, 1].
        ssim_weight: Weight for SSIM term relative to L1.

    Returns:
        Scalar photometric loss.
    """
    # L1 loss
    l1 = torch.abs(rendered - ground_truth).mean()

    # SSIM loss (simplified window-based)
    ssim_val = _compute_ssim(rendered, ground_truth)
    ssim_loss = 1.0 - ssim_val

    return l1 + ssim_weight * ssim_loss


def _compute_ssim(
    x: Tensor,
    y: Tensor,
    window_size: int = 11,
    channel: int = 3,
) -> Tensor:
    """Compute SSIM between two images.

    Uses a Gaussian window for local statistics computation.
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Create Gaussian window
    sigma = 1.5
    gauss = torch.arange(window_size, dtype=torch.float32, device=x.device)
    gauss = torch.exp(-((gauss - window_size // 2) ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()

    # 2D window via outer product
    window_1d = gauss.unsqueeze(1)  # (W, 1)
    window_2d = window_1d @ window_1d.t()  # (W, W)
    window_2d = window_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, W, W)
    window = window_2d.expand(channel, 1, -1, -1).contiguous()  # (C, 1, W, W)

    padding = window_size // 2

    mu_x = F.conv2d(x, window, padding=padding, groups=channel)
    mu_y = F.conv2d(y, window, padding=padding, groups=channel)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, window, padding=padding, groups=channel) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, window, padding=padding, groups=channel) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=padding, groups=channel) - mu_xy

    numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)

    ssim_map = numerator / denominator
    return ssim_map.mean()


# ---------------------------------------------------------------------------
# Total Loss with Three-Stage Schedule
# ---------------------------------------------------------------------------

def compute_total_loss(
    # Gaussian parameters
    gaussian_means: Tensor,
    predicted_occupancy: Tensor,
    semantic_logits: Optional[Tensor] = None,
    # VGGT supervision
    vggt_points: Optional[Tensor] = None,
    occupied_mask: Optional[Tensor] = None,
    free_mask: Optional[Tensor] = None,
    semantic_labels: Optional[Tensor] = None,
    # Rendered images (auxiliary)
    rendered_images: Optional[Tensor] = None,
    gt_images: Optional[Tensor] = None,
    # Stage weights
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    delta: float = 0.3,
    eta: float = 0.1,
    # Loss hyperparameters
    focal_gamma: float = 2.0,
    hinge_epsilon: float = 0.05,
    ssim_weight: float = 0.2,
    label_smoothing: float = 0.0,
    use_bce: bool = False,
) -> dict[str, Tensor]:
    """Compute total loss with per-component breakdown.

    L_total = α·L_depth + β·L_occ + γ·L_free + δ·L_sem + η·L_color

    Args:
        gaussian_means: (N, 3) Gaussian centers.
        predicted_occupancy: (N,) predicted occupancy values.
        semantic_logits: (N, K) semantic class logits.
        vggt_points: (M, 3) VGGT surface points.
        occupied_mask: (N,) bool, occupied Gaussians.
        free_mask: (N,) bool, free-space Gaussians.
        semantic_labels: (N,) integer semantic labels.
        rendered_images: (B, 3, H, W) rendered images.
        gt_images: (B, 3, H, W) ground truth images.
        alpha, beta, gamma, delta, eta: Loss weights per stage.
        focal_gamma: Focal loss γ parameter.
        hinge_epsilon: Hinge loss ε threshold.
        ssim_weight: SSIM term weight in color loss.
        label_smoothing: Label smoothing for semantic CE.

    Returns:
        Dict with 'total' and individual loss components.
    """
    losses = {}
    total = torch.tensor(0.0, device=gaussian_means.device)

    # L_depth: Chamfer distance
    if vggt_points is not None and alpha > 0:
        l_depth = chamfer_depth_loss(gaussian_means, vggt_points, occupied_mask)
        losses["depth"] = l_depth
        total = total + alpha * l_depth

    # L_occ: Focal or BCE occupancy loss
    if occupied_mask is not None and free_mask is not None and beta > 0:
        labeled_mask = occupied_mask | free_mask
        if labeled_mask.any():
            labels = occupied_mask.float()
            if use_bce:
                # Standard BCE (ablation: no focal modulation)
                eps = 1e-6
                o = predicted_occupancy[labeled_mask].clamp(eps, 1 - eps)
                l_occ = F.binary_cross_entropy(o, labels[labeled_mask], reduction="mean")
            else:
                l_occ = focal_occupancy_loss(
                    predicted_occupancy[labeled_mask],
                    labels[labeled_mask],
                    gamma=focal_gamma,
                )
            losses["occupancy"] = l_occ
            total = total + beta * l_occ

    # L_free: Free-space hinge loss
    if free_mask is not None and gamma > 0:
        l_free = hinge_free_space_loss(
            predicted_occupancy, free_mask, epsilon=hinge_epsilon
        )
        losses["free_space"] = l_free
        total = total + gamma * l_free

    # L_semantic: Semantic cross-entropy
    if (
        semantic_logits is not None
        and semantic_labels is not None
        and occupied_mask is not None
        and delta > 0
    ):
        l_sem = semantic_cross_entropy_loss(
            semantic_logits, semantic_labels, occupied_mask,
            label_smoothing=label_smoothing,
        )
        losses["semantic"] = l_sem
        total = total + delta * l_sem

    # L_color: Auxiliary photometric loss
    if rendered_images is not None and gt_images is not None and eta > 0:
        l_color = color_reconstruction_loss(rendered_images, gt_images, ssim_weight)
        losses["color"] = l_color
        total = total + eta * l_color

    losses["total"] = total
    return losses


# ---------------------------------------------------------------------------
# Three-Stage Training Schedule
# ---------------------------------------------------------------------------

STAGE_SCHEDULE = {
    1: {  # Warmup (~30% iterations): move Gaussians to surfaces
        "alpha": 1.0,
        "beta": 0.3,
        "gamma": 0.1,
        "delta": 0.0,
        "eta": 0.3,
    },
    2: {  # Main (~50% iterations): occupancy + free-space + semantics
        "alpha": 0.5,
        "beta": 1.0,
        "gamma": 0.5,
        "delta": 0.3,
        "eta": 0.1,
    },
    3: {  # Fine-tune (~20% iterations): refine, color exits
        "alpha": 0.3,
        "beta": 1.0,
        "gamma": 1.0,
        "delta": 0.5,
        "eta": 0.05,
    },
}


def get_stage_weights(
    current_step: int,
    total_steps: int,
    schedule: Optional[dict] = None,
) -> dict[str, float]:
    """Get loss weights for current training step.

    Implements the three-stage training schedule from Phase B design §2.5:
        Stage 1 (0-30%): α=1.0, β=0.3, γ=0.1, δ=0, η=0.3
        Stage 2 (30-80%): α=0.5, β=1.0, γ=0.5, δ=0.3, η=0.1
        Stage 3 (80-100%): α=0.3, β=1.0, γ=1.0, δ=0.5, η=0.05

    Args:
        current_step: Current training iteration.
        total_steps: Total training iterations.
        schedule: Optional custom schedule override.

    Returns:
        Dict with keys alpha, beta, gamma, delta, eta.
    """
    if schedule is None:
        schedule = STAGE_SCHEDULE

    progress = current_step / max(total_steps, 1)

    if progress < 0.3:
        stage = 1
    elif progress < 0.8:
        stage = 2
    else:
        stage = 3

    return schedule[stage].copy()

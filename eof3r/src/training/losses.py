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
    predicted_occupancy: Optional[Tensor] = None,
    max_gaussians: int = 4096,
    max_points: int = 2048,
) -> Tensor:
    """Bidirectional Chamfer Distance between Gaussian means and VGGT surface points.

    From Phase B design §2.4:
        L_depth = (1/|P|) Σ_p min_i ||μ_i - p||² + (1/|O|) Σ_{i∈O} min_p ||μ_i - p||²

    Forward term: each VGGT surface point needs at least one Gaussian nearby.
    Backward term: each occupied Gaussian needs to be near some VGGT surface
    point — penalizes floaters. When predicted_occupancy is provided, the
    backward term is occupancy-weighted: high-occupancy Gaussians are pulled
    harder to surfaces, low-occupancy Gaussians contribute less.

    Args:
        gaussian_means: (N, 3) Gaussian center positions in world coords.
        vggt_points: (M, 3) VGGT surface point cloud.
        occupied_mask: (N,) bool mask for occupied Gaussians. If None and
            predicted_occupancy is None, all Gaussians are used in backward term.
        predicted_occupancy: (N,) predicted occupancy values [0,1]. When provided,
            backward term weights by occupancy (detached to avoid circular gradients).
            Takes precedence over occupied_mask for weighting.
        max_gaussians: Subsample Gaussians if N exceeds this (for memory).
        max_points: Subsample surface points if M exceeds this (for memory).

    Returns:
        Scalar chamfer distance loss.
    """
    if gaussian_means.ndim != 2 or vggt_points.ndim != 2:
        raise ValueError(f"Expected 2D tensors, got {gaussian_means.shape}, {vggt_points.shape}")
    if gaussian_means.shape[1] != 3 or vggt_points.shape[1] != 3:
        raise ValueError(f"Expected 3D points, got {gaussian_means.shape[1]}, {vggt_points.shape[1]}")

    # Subsample for memory efficiency
    N = gaussian_means.shape[0]
    M = vggt_points.shape[0]
    device = gaussian_means.device

    if M > max_points:
        idx = torch.randperm(M, device=device)[:max_points]
        vggt_points = vggt_points[idx]

    # Forward: for each VGGT point, find nearest Gaussian
    # Use chunked computation for large N
    if N > max_gaussians:
        # Random subsample of Gaussians for forward term
        idx = torch.randperm(N, device=device)[:max_gaussians]
        means_fwd = gaussian_means[idx]
    else:
        means_fwd = gaussian_means

    diff_fwd = vggt_points.unsqueeze(1) - means_fwd.unsqueeze(0)  # (M, N', 3)
    dist_fwd = (diff_fwd ** 2).sum(dim=-1)  # (M, N')
    min_dist_fwd = dist_fwd.min(dim=1).values  # (M,)
    loss_fwd = min_dist_fwd.mean()

    # Backward: for each Gaussian, find nearest VGGT point
    # Determine weights and which Gaussians to include
    if predicted_occupancy is not None:
        # Occupancy-weighted: use all Gaussians, weight by predicted occupancy
        # detach() prevents circular gradient: occupancy weights guide Chamfer,
        # but Chamfer doesn't backprop through occupancy head
        weights_bwd = predicted_occupancy.detach().clamp(min=0.0)
        means_bwd = gaussian_means
    elif occupied_mask is not None:
        weights_bwd = occupied_mask.float()
        means_bwd = gaussian_means
    else:
        weights_bwd = torch.ones(N, device=device)
        means_bwd = gaussian_means

    # Subsample for backward term
    if means_bwd.shape[0] > max_gaussians:
        idx = torch.randperm(means_bwd.shape[0], device=device)[:max_gaussians]
        means_bwd = means_bwd[idx]
        weights_bwd = weights_bwd[idx]

    if means_bwd.shape[0] == 0:
        return loss_fwd

    diff_bwd = means_bwd.unsqueeze(1) - vggt_points.unsqueeze(0)  # (|O|', M, 3)
    dist_bwd = (diff_bwd ** 2).sum(dim=-1)  # (|O|', M)
    min_dist_bwd = dist_bwd.min(dim=1).values  # (|O|',)

    # Weighted mean: high-occupancy Gaussians penalized more for being far
    weight_sum = weights_bwd.sum().clamp(min=1)
    loss_bwd = (min_dist_bwd * weights_bwd).sum() / weight_sum

    return loss_fwd + loss_bwd


# ---------------------------------------------------------------------------
# L_position: Occupancy-Guided Position Loss
# ---------------------------------------------------------------------------

def occupancy_guided_position_loss(
    gaussian_means: Tensor,
    predicted_occupancy: Tensor,
    vggt_points: Tensor,
    kappa_attract: float = 1.0,
    kappa_repel: float = 0.1,
    free_surface_margin: float = 0.1,
    max_gaussians: int = 4096,
    max_points: int = 2048,
) -> Tensor:
    """Directly move Gaussians based on occupancy predictions.

    Provides a DIRECT gradient path from occupancy to positions, bypassing
    the MLP bottleneck in the occupancy head.

    - High-occupancy Gaussians (occ > 0.5): attracted toward nearest VGGT
      surface point — penalizes distance to surface.
    - Low-occupancy Gaussians (occ ≤ 0.5): repelled from VGGT surface —
      only penalized if TOO close to surface (within free_surface_margin).

    The key insight: instead of asking occupancy loss to move positions
    through the thin MLP, we compute position targets based on occupancy
    predictions and compare current positions to those targets directly.

    Uses detach() on occupancy threshold to avoid circular gradients:
    the occupancy value determines WHICH Gaussians are attracted/repelled,
    but the position gradient goes directly to means (not through occ head).

    Args:
        gaussian_means: (N, 3) Gaussian center positions.
        predicted_occupancy: (N,) predicted occupancy values [0, 1].
        vggt_points: (M, 3) VGGT surface points.
        kappa_attract: Weight for attraction term (occupied → surface).
        kappa_repel: Weight for repulsion term (free → away from surface).
        free_surface_margin: Minimum distance free Gaussians must maintain
            from surfaces (meters). Closer Gaussians are repelled.
        max_gaussians: Subsample Gaussians for memory.
        max_points: Subsample surface points for memory.

    Returns:
        Scalar position guidance loss.
    """
    N = gaussian_means.shape[0]
    M = vggt_points.shape[0]
    device = gaussian_means.device

    # Subsample surface points for memory
    if M > max_points:
        idx = torch.randperm(M, device=device)[:max_points]
        pts = vggt_points[idx]
    else:
        pts = vggt_points

    # Hard assignment based on occupancy threshold
    # detach() so occupancy only determines grouping, not gradient source
    occ_mask = (predicted_occupancy > 0.5).detach()
    free_mask = ~occ_mask

    loss = torch.tensor(0.0, device=device)

    # Occupied term: attract to nearest surface point
    if occ_mask.any() and kappa_attract > 0:
        means_occ = gaussian_means[occ_mask]
        if means_occ.shape[0] > max_gaussians:
            idx = torch.randperm(means_occ.shape[0], device=device)[:max_gaussians]
            means_occ = means_occ[idx]

        # Pairwise distances: (|O|, M)
        dists_occ = torch.cdist(means_occ, pts, p=2.0)  # L2 distance
        min_dists = dists_occ.min(dim=1).values  # (|O|,)
        loss = loss + kappa_attract * min_dists.mean()

    # Free term: repel Gaussians that are too close to surfaces
    if free_mask.any() and kappa_repel > 0:
        means_free = gaussian_means[free_mask]
        if means_free.shape[0] > max_gaussians:
            idx = torch.randperm(means_free.shape[0], device=device)[:max_gaussians]
            means_free = means_free[idx]

        dists_free = torch.cdist(means_free, pts, p=2.0)  # (|F|, M)
        min_dists = dists_free.min(dim=1).values  # (|F|,)

        # Hinge: only penalize if closer than margin to surface
        violation = torch.clamp(free_surface_margin - min_dists, min=0.0)
        loss = loss + kappa_repel * violation.mean()

    return loss


# ---------------------------------------------------------------------------
# L_bev: BEV Coverage Loss (differentiable)
# ---------------------------------------------------------------------------

def bev_coverage_loss(
    bev_grid: Tensor,
    target_coverage: float = 0.10,
    threshold: float = 0.3,
) -> Tensor:
    """Penalize low BEV coverage — encourages occupied Gaussians to spread.

    Provides direct gradient signal: if BEV is too empty, move Gaussians
    to produce coverage. This connects the BEV output back to Gaussian
    positions through the differentiable BEV projection.

    L_bev = max(0, target_coverage - actual_coverage)²

    Args:
        bev_grid: (H, W) BEV occupancy grid with values in [0, 1].
        target_coverage: Minimum desired coverage fraction.
        threshold: Occupancy threshold for "occupied" cells.

    Returns:
        Scalar coverage loss (0 if coverage ≥ target).
    """
    actual = (bev_grid > threshold).float().mean()
    shortfall = torch.clamp(target_coverage - actual, min=0.0)
    return shortfall ** 2


# ---------------------------------------------------------------------------
# Differentiable BEV Projection (torch-based, for training)
# ---------------------------------------------------------------------------

def differentiable_bev_projection(
    means_zup: Tensor,
    occupancies: Tensor,
    bev_resolution: float = 0.05,
    bev_range: tuple[float, float, float, float] = (-10.0, -10.0, 10.0, 10.0),
    height_range: tuple[float, float] = (-0.5, 2.0),
    gaussian_sigma_cells: float = 2.0,
) -> Tensor:
    """Torch-differentiable BEV projection for training.

    Projects 3D Gaussians to a 2D BEV grid using their occupancy values.
    Fully differentiable — gradients flow from BEV back to occupancy
    values and Gaussian positions.

    Uses a soft scatter with Gaussian kernel falloff per cell, enabling
    gradient propagation through the grid.

    Args:
        means_zup: (N, 3) Gaussian centers in Z-up coords (X-fwd, Y-left, Z-up).
        occupancies: (N,) predicted occupancy values [0, 1].
        bev_resolution: Meters per grid cell.
        bev_range: (x_min, y_min, x_max, y_max) in meters.
        height_range: (z_min, z_max) height filter in Z-up.
        gaussian_sigma_cells: Sigma of Gaussian kernel in cell units for
            per-point smoothing (replaces the fixed global sigma).

    Returns:
        (H, W) BEV occupancy grid [0, 1], differentiable.
    """
    x_min, y_min, x_max, y_max = bev_range
    z_min, z_max = height_range

    # Height filter
    z = means_zup[:, 2]
    height_mask = (z >= z_min) & (z <= z_max)
    m = means_zup[height_mask]
    o = occupancies[height_mask]

    if m.shape[0] == 0:
        H = int((y_max - y_min) / bev_resolution)
        W = int((x_max - x_min) / bev_resolution)
        return torch.zeros(H, W, device=means_zup.device, dtype=means_zup.dtype)

    # Map to grid cell indices (continuous, for soft assignment)
    x_cells = (m[:, 0] - x_min) / bev_resolution  # (N',)
    y_cells = (m[:, 1] - y_min) / bev_resolution  # (N',)

    H = int((y_max - y_min) / bev_resolution)
    W = int((x_max - x_min) / bev_resolution)

    # Create grid of cell centers
    device = means_zup.device
    dtype = means_zup.dtype
    gy = torch.arange(H, device=device, dtype=dtype)  # (H,)
    gx = torch.arange(W, device=device, dtype=dtype)  # (W,)

    # For each Gaussian, compute soft weight to each nearby cell
    # Use a Gaussian kernel: w_ij = exp(-dist² / (2·σ²))
    sigma2 = 2.0 * gaussian_sigma_cells ** 2

    # Expand for pairwise computation: (N', H, W)
    dy = y_cells.unsqueeze(-1).unsqueeze(-1) - gy.unsqueeze(0).unsqueeze(-1)  # (N', H, 1)
    dx = x_cells.unsqueeze(-1).unsqueeze(-1) - gx.unsqueeze(0).unsqueeze(0)  # (N', 1, W)
    dist2 = dy.pow(2) + dx.pow(2)  # (N', H, W)

    # Truncate beyond 3·sigma
    max_dist2 = (3.0 * gaussian_sigma_cells) ** 2
    weight = torch.exp(-dist2 / sigma2)  # (N', H, W)
    weight = weight * (dist2 <= max_dist2)  # truncate

    # Weighted sum: each Gaussian contributes occupancy × weight
    weighted_occ = o.unsqueeze(-1).unsqueeze(-1) * weight  # (N', H, W)
    bev = weighted_occ.sum(dim=0)  # (H, W)

    # Normalize: soft count of Gaussians per cell (prevent over-saturation)
    count = weight.sum(dim=0).clamp(min=1.0)  # (H, W)
    bev = bev / count

    return bev.clamp(0.0, 1.0)

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
    # BEV grid (from differentiable projection)
    bev_grid: Optional[Tensor] = None,
    # Stage weights
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    delta: float = 0.3,
    eta: float = 0.1,
    theta: float = 0.0,
    zeta: float = 0.0,
    # Loss hyperparameters
    focal_gamma: float = 2.0,
    hinge_epsilon: float = 0.05,
    ssim_weight: float = 0.2,
    label_smoothing: float = 0.0,
    use_bce: bool = False,
    # Position loss hyperparameters
    pos_kappa_attract: float = 1.0,
    pos_kappa_repel: float = 0.1,
    # BEV loss hyperparameters
    bev_target_coverage: float = 0.10,
) -> dict[str, Tensor]:
    """Compute total loss with per-component breakdown.

    Extended loss:
        L_total = α·L_depth + β·L_occ + γ·L_free + δ·L_sem + η·L_color
                + θ·L_position + ζ·L_bev

    Where:
        L_position: Occupancy-guided position loss (direct gradient path)
        L_bev: BEV coverage loss (connects BEV quality to occupancy)

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
        bev_grid: (H, W) differentiable BEV grid from bev_projection.
        alpha, beta, gamma, delta, eta: Classic loss weights per stage.
        theta: Weight for occupancy-guided position loss (L_position).
        zeta: Weight for BEV coverage loss (L_bev).
        focal_gamma: Focal loss γ parameter.
        hinge_epsilon: Hinge loss ε threshold.
        ssim_weight: SSIM term weight in color loss.
        label_smoothing: Label smoothing for semantic CE.
        use_bce: Use BCE instead of focal loss for occupancy.
        pos_kappa_attract: Attraction weight for position loss.
        pos_kappa_repel: Repulsion weight for position loss.
        bev_target_coverage: Target BEV coverage fraction.

    Returns:
        Dict with 'total' and individual loss components.
    """
    losses = {}
    total = torch.tensor(0.0, device=gaussian_means.device)

    # L_depth: Chamfer distance (now occupancy-weighted)
    if vggt_points is not None and alpha > 0:
        l_depth = chamfer_depth_loss(
            gaussian_means, vggt_points, occupied_mask,
            predicted_occupancy=predicted_occupancy,
        )
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

    # L_position: Occupancy-guided position loss (direct gradient path)
    if vggt_points is not None and theta > 0:
        l_position = occupancy_guided_position_loss(
            gaussian_means, predicted_occupancy, vggt_points,
            kappa_attract=pos_kappa_attract,
            kappa_repel=pos_kappa_repel,
        )
        losses["position"] = l_position
        total = total + theta * l_position

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

    # L_bev: BEV coverage loss
    if bev_grid is not None and zeta > 0:
        l_bev = bev_coverage_loss(bev_grid, target_coverage=bev_target_coverage)
        losses["bev"] = l_bev
        total = total + zeta * l_bev

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
        "theta": 0.5,
        "zeta": 0.3,
    },
    2: {  # Main (~50% iterations): occupancy + free-space + semantics
        "alpha": 0.5,
        "beta": 1.0,
        "gamma": 0.5,
        "delta": 0.3,
        "eta": 0.1,
        "theta": 0.3,
        "zeta": 0.1,
    },
    3: {  # Fine-tune (~20% iterations): refine, color exits
        "alpha": 0.3,
        "beta": 1.0,
        "gamma": 1.0,
        "delta": 0.5,
        "eta": 0.05,
        "theta": 0.1,
        "zeta": 0.05,
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

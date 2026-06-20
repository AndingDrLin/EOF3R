"""VGGT-based per-Gaussian supervision labeling.

Implements the projection-based labeling from Phase B design §2.3:
    For each Gaussian i:
    1. Project to VGGT camera frame: μ̃_i = T_v^{-1} · μ_i
    2. Pixel coordinates: (u_i, v_i) = π_K(μ̃_i)
    3. Compare depth: Δd_i = μ̃_i^z - D^vggt(u_i, v_i)
    4. Adaptive threshold: σ_i = κ · max_eig(Σ_i)

    Label:
        y_i = 1 (OCCUPIED)   if |Δd_i| ≤ σ_i
        y_i = 0 (FREE)       if Δd_i < -σ_i
        y_i = mask (UNKNOWN) if Δd_i > σ_i

This was validated in Phase A.1 POC (2026-06-19):
    - 2.6% Gaussians labeled as occupied (near VGGT surface)
    - 68.9% labeled as free (in front of surface)
    - 28.5% unknown (behind surface, occluded)
"""

from __future__ import annotations

import torch
from torch import Tensor
from dataclasses import dataclass
from typing import Optional


@dataclass
class GaussianLabels:
    """Per-Gaussian supervision labels from VGGT projection.

    Attributes:
        occupied: (N,) bool mask for occupied Gaussians (near VGGT surface).
        free: (N,) bool mask for free-space Gaussians (in front of surface).
        unknown: (N,) bool mask for unknown Gaussians (behind surface).
        depth_error: (N,) signed depth difference Δd_i.
        threshold: (N,) adaptive threshold σ_i per Gaussian.
        semantic_labels: (N,) integer semantic labels from SAM2/YOLO (if available).
    """
    occupied: Tensor
    free: Tensor
    unknown: Tensor
    depth_error: Tensor
    threshold: Tensor
    semantic_labels: Optional[Tensor] = None

    @property
    def num_occupied(self) -> int:
        return int(self.occupied.sum().item())

    @property
    def num_free(self) -> int:
        return int(self.free.sum().item())

    @property
    def num_unknown(self) -> int:
        return int(self.unknown.sum().item())

    @property
    def num_labeled(self) -> int:
        return self.num_occupied + self.num_free

    def summary(self) -> str:
        total = self.occupied.shape[0]
        return (
            f"GaussianLabels(total={total}, "
            f"occupied={self.num_occupied} ({100*self.num_occupied/total:.1f}%), "
            f"free={self.num_free} ({100*self.num_free/total:.1f}%), "
            f"unknown={self.num_unknown} ({100*self.num_unknown/total:.1f}%))"
        )


def label_gaussians_by_vggt_projection(
    gaussian_means: Tensor,
    gaussian_covariances: Tensor,
    vggt_depth: Tensor,
    camera_extrinsics: Tensor,
    camera_intrinsics: Tensor,
    kappa: float = 3.0,
    semantic_labels_2d: Optional[Tensor] = None,
) -> GaussianLabels:
    """Label each Gaussian as occupied/free/unknown via VGGT depth projection.

    From Phase B design §2.3, this is the core supervision mechanism:
    1. Project Gaussian centers to VGGT camera frame
    2. Query VGGT depth at projected pixel
    3. Compare Gaussian depth vs VGGT depth with adaptive threshold

    Args:
        gaussian_means: (N, 3) Gaussian centers in world coords.
        gaussian_covariances: (N, 3, 3) Gaussian covariance matrices.
        vggt_depth: (H, W) VGGT depth map in camera frame.
        camera_extrinsics: (4, 4) world-from-camera transform (T_w_c).
        camera_intrinsics: (3, 3) camera intrinsic matrix K.
        kappa: Multiplier for adaptive threshold (σ_i = κ · max_eig(Σ_i)).
            Higher κ = more tolerant of depth uncertainty.
        semantic_labels_2d: (H, W) optional 2D semantic labels to project
            onto Gaussians via nearest-pixel lookup.

    Returns:
        GaussianLabels with occupied/free/unknown masks.
    """
    device = gaussian_means.device
    N = gaussian_means.shape[0]
    H, W = vggt_depth.shape

    # Step 1: Transform Gaussians to camera frame
    # T_c_w = T_w_c^{-1}
    T_c_w = torch.inverse(camera_extrinsics)  # (4, 4)
    R_c_w = T_c_w[:3, :3]  # (3, 3)
    t_c_w = T_c_w[:3, 3]   # (3,)

    # μ̃_i = R_c_w · μ_i + t_c_w
    means_cam = (R_c_w @ gaussian_means.T).T + t_c_w  # (N, 3)

    # Step 2: Project to pixel coordinates
    # (u, v) = K · μ̃ / μ̃_z
    K = camera_intrinsics  # (3, 3)

    # Project
    projected = (K @ means_cam.T).T  # (N, 3)
    depths_cam = projected[:, 2]  # (N,) z-depth in camera frame
    pixels = projected[:, :2] / depths_cam.unsqueeze(-1).clamp(min=1e-6)  # (N, 2)

    u = pixels[:, 0].long()  # (N,)
    v = pixels[:, 1].long()  # (N,)

    # Step 3: Check bounds and query VGGT depth
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (depths_cam > 0)

    # Initialize all as unknown
    occupied = torch.zeros(N, dtype=torch.bool, device=device)
    free = torch.zeros(N, dtype=torch.bool, device=device)
    unknown = torch.ones(N, dtype=torch.bool, device=device)
    depth_error = torch.zeros(N, device=device)
    threshold = torch.zeros(N, device=device)

    if not in_bounds.any():
        return GaussianLabels(
            occupied=occupied, free=free, unknown=unknown,
            depth_error=depth_error, threshold=threshold,
        )

    # Get valid Gaussians
    valid_idx = in_bounds.nonzero(as_tuple=True)[0]
    valid_u = u[valid_idx]
    valid_v = v[valid_idx]
    valid_depth = depths_cam[valid_idx]

    # Query VGGT depth at projected pixels
    vggt_d = vggt_depth[valid_v, valid_u]  # (|valid|,)

    # Step 4: Compute depth error and adaptive threshold
    # Δd_i = μ̃_i^z - D^vggt(u_i, v_i)
    delta_d = valid_depth - vggt_d  # (|valid|,)

    # σ_i = κ · sqrt(max_eig(Σ_i))
    # Use eigvalsh for proper eigenvalue computation (3x3 is cheap)
    valid_cov = gaussian_covariances[valid_idx]  # (|valid|, 3, 3)
    eigvals = torch.linalg.eigvalsh(valid_cov)  # (|valid|, 3), ascending
    max_eig = eigvals[:, -1]  # (|valid|,) — largest eigenvalue = variance along major axis
    sigma = kappa * torch.sqrt(max_eig.clamp(min=1e-12))

    # Step 5: Apply labeling rules
    # y_i = 1 (OCCUPIED)   if |Δd_i| ≤ σ_i
    # y_i = 0 (FREE)       if Δd_i < -σ_i
    # y_i = mask (UNKNOWN) if Δd_i > σ_i
    label_occupied = delta_d.abs() <= sigma
    label_free = delta_d < -sigma
    label_unknown = delta_d > sigma

    # Write back to full arrays
    occupied[valid_idx] = label_occupied
    free[valid_idx] = label_free
    unknown[valid_idx] = label_unknown
    depth_error[valid_idx] = delta_d
    threshold[valid_idx] = sigma

    # Optional: project 2D semantic labels to Gaussians
    sem_labels = None
    if semantic_labels_2d is not None:
        sem_labels = torch.zeros(N, dtype=torch.long, device=device)
        valid_sem = semantic_labels_2d[valid_v, valid_u]
        sem_labels[valid_idx] = valid_sem

    return GaussianLabels(
        occupied=occupied,
        free=free,
        unknown=unknown,
        depth_error=depth_error,
        threshold=threshold,
        semantic_labels=sem_labels,
    )


def compute_vggt_surface_points(
    vggt_depth: Tensor,
    camera_intrinsics: Tensor,
    camera_extrinsics: Tensor,
    subsample: int = 1,
) -> Tensor:
    """Convert VGGT depth map to 3D surface point cloud.

    Used as supervision for the Chamfer distance loss (L_depth).

    Args:
        vggt_depth: (H, W) depth map in camera frame.
        camera_intrinsics: (3, 3) intrinsic matrix K.
        camera_extrinsics: (4, 4) world-from-camera transform.
        subsample: Subsample factor for point density reduction.

    Returns:
        (M, 3) surface points in world coordinates.
    """
    H, W = vggt_depth.shape
    device = vggt_depth.device

    # Create pixel grid
    v_coords, u_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )

    # Subsample
    if subsample > 1:
        v_coords = v_coords[::subsample, ::subsample]
        u_coords = u_coords[::subsample, ::subsample]
        depth_sub = vggt_depth[::subsample, ::subsample]
    else:
        depth_sub = vggt_depth

    # Flatten
    u_flat = u_coords.reshape(-1)
    v_flat = v_coords.reshape(-1)
    d_flat = depth_sub.reshape(-1)

    # Filter valid depths
    valid = d_flat > 0
    u_valid = u_flat[valid]
    v_valid = v_flat[valid]
    d_valid = d_flat[valid]

    # Back-project to camera frame
    K_inv = torch.inverse(camera_intrinsics)
    pixels_homo = torch.stack([u_valid, v_valid, torch.ones_like(u_valid)], dim=-1)  # (M, 3)
    points_cam = (K_inv @ pixels_homo.T).T * d_valid.unsqueeze(-1)  # (M, 3)

    # Transform to world frame
    R_w_c = camera_extrinsics[:3, :3]
    t_w_c = camera_extrinsics[:3, 3]
    points_world = (R_w_c @ points_cam.T).T + t_w_c  # (M, 3)

    return points_world


def merge_multi_view_labels(
    labels_list: list[GaussianLabels],
) -> GaussianLabels:
    """Merge supervision labels from multiple camera views.

    A Gaussian is OCCUPIED if any view labels it as occupied.
    A Gaussian is FREE only if ALL views that see it label it as free.
    A Gaussian is UNKNOWN if no view sees it, or if labels conflict.

    Args:
        labels_list: List of GaussianLabels from different views.

    Returns:
        Merged GaussianLabels.
    """
    if len(labels_list) == 0:
        raise ValueError("Empty labels list")

    if len(labels_list) == 1:
        return labels_list[0]

    N = labels_list[0].occupied.shape[0]
    device = labels_list[0].occupied.device

    # Accumulate votes
    occupied_votes = torch.zeros(N, dtype=torch.long, device=device)
    free_votes = torch.zeros(N, dtype=torch.long, device=device)
    total_votes = torch.zeros(N, dtype=torch.long, device=device)

    for labels in labels_list:
        occupied_votes += labels.occupied.long()
        free_votes += labels.free.long()
        # Count how many views saw this Gaussian (not unknown)
        total_votes += (labels.occupied | labels.free).long()

    # Merge rules:
    # - OCCUPIED if any view says occupied
    # - FREE if all views that see it say free (and at least one sees it)
    # - UNKNOWN otherwise
    occupied = occupied_votes > 0
    free = (free_votes > 0) & (occupied_votes == 0) & (total_votes > 0)
    unknown = ~occupied & ~free

    # Use the first view's depth_error and threshold as representative
    # (could also average, but first is simplest)
    return GaussianLabels(
        occupied=occupied,
        free=free,
        unknown=unknown,
        depth_error=labels_list[0].depth_error,
        threshold=labels_list[0].threshold,
        semantic_labels=labels_list[0].semantic_labels,
    )

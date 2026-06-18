#!/usr/bin/env python3
"""Minimal occupancy-head experiment: VGGT-supervised per-Gaussian occupancy.

PHASE B — Proof of Concept:
  1. Run MVSplat → Gaussians with rendering opacity (baseline).
  2. Run VGGT → pointmap → KD-tree.
  3. Assign per-Gaussian binary occupancy labels from VGGT nearest-neighbor.
  4. Train a tiny MLP (7→16→8→1) to predict occupancy from Gaussian features.
  5. Compare three BEV projections:
     (a) opacity-based  (baseline — confirmed broken by Phase A)
     (b) VGGT-label-based  (oracle upper bound)
     (c) MLP-predicted  (learnable occupancy head)

This directly tests the hypothesis: "If we replace rendering opacity with
geometry-supervised occupancy, does BEV become usable?"

Usage:
  source ~/anaconda3/etc/profile.d/conda.sh && conda activate eof3r
  python eof3r/scripts/eval/test_occupancy_head.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_CODE_ROOT = _REPO_ROOT / "eof3r"
_MVSPLAT_ROOT = _REPO_ROOT / "baselines" / "mvsplat"
_PUBLIC_DIR = _REPO_ROOT / "data" / "public" / "re10k_samples"
_OUTPUT_DIR = _REPO_ROOT / "outputs" / "eval" / "occupancy_head"
sys.path.insert(0, str(_CODE_ROOT))


# ---------------------------------------------------------------------------
# Occupancy Head MLP
# ---------------------------------------------------------------------------

class OccupancyHead(nn.Module):
    """Tiny MLP: per-Gaussian features → occupancy probability [0,1]."""

    def __init__(self, in_dim: int = 7, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# BEV projection (mirrors bev_projector.py logic but self-contained)
# ---------------------------------------------------------------------------

def project_to_bev(
    means: np.ndarray,       # (N, 3) Y-up
    occupancies: np.ndarray,  # (N,)  [0,1]
    scales: np.ndarray,       # (N, 3)
    height_range: tuple = (-1.0, 8.0),
    grid_res: float = 0.05,
    x_range: tuple | None = None,
    z_range: tuple | None = None,
) -> np.ndarray:
    """Project Gaussians to BEV (XZ plane) using occupancy instead of opacity."""
    mask = (means[:, 1] >= height_range[0]) & (means[:, 1] <= height_range[1])
    m = means[mask]
    o = occupancies[mask]
    s = scales[mask]

    if len(m) == 0:
        return np.zeros((100, 100), dtype=np.float32)

    # Auto bounds (+20% margin).
    if x_range is None:
        x_margin = (m[:, 0].max() - m[:, 0].min()) * 0.2 + 0.5
        x_min, x_max = float(m[:, 0].min() - x_margin), float(m[:, 0].max() + x_margin)
    else:
        x_min, x_max = x_range
    if z_range is None:
        z_margin = (m[:, 2].max() - m[:, 2].min()) * 0.2 + 0.5
        z_min, z_max = float(m[:, 2].min() - z_margin), float(m[:, 2].max() + z_margin)
    else:
        z_min, z_max = z_range

    n_x = max(1, int((x_max - x_min) / grid_res))
    n_z = max(1, int((z_max - z_min) / grid_res))
    bev = np.zeros((n_z, n_x), dtype=np.float32)

    for i in range(len(m)):
        x, _y, z = m[i]
        sx, _sy, sz = s[i]
        col = int((x - x_min) / grid_res)
        row = int((z - z_min) / grid_res)
        if 0 <= row < n_z and 0 <= col < n_x:
            r = max(sx, sz) * 3.0
            rc = max(1, int(r / grid_res))
            for dr in range(-rc, rc + 1):
                for dc in range(-rc, rc + 1):
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < n_z and 0 <= nc < n_x:
                        dist = np.sqrt(dr**2 + dc**2) * grid_res
                        if dist <= r:
                            w = o[i] * np.exp(-0.5 * (dist / max(r, 1e-6)) ** 2)
                            bev[nr, nc] = max(bev[nr, nc], w)

    print(f"  BEV: {len(m)} Gaussians → ({n_z}, {n_x}) grid, "
          f"occ>(0.3)={(bev > 0.3).sum()}")
    return bev


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_bev_metrics(bev: np.ndarray, prefix: str = "bev") -> dict:
    """Coverage, density, extent for a BEV grid."""
    total = bev.size
    out = {}
    for t in [0.1, 0.3, 0.5, 0.7]:
        out[f"{prefix}_coverage_t{t}".replace(".", "_")] = round(
            float((bev > t).sum()) / max(total, 1), 6
        )
    occ_mask = bev > 0.01
    out[f"{prefix}_density"] = round(
        float(bev[occ_mask].mean()) if occ_mask.sum() > 0 else 0.0, 6
    )
    out[f"{prefix}_extent_m2"] = round(
        float(occ_mask.sum()) * 0.05 * 0.05, 4
    )
    out[f"{prefix}_occupied_cells"] = int((bev > 0.3).sum())
    out[f"{prefix}_max"] = round(float(bev.max()), 4)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("Occupancy Head Experiment — Phase B Proof of Concept")
    print("=" * 70)

    # ---- Load images ----
    images_np = []
    for i in range(4):
        p = _PUBLIC_DIR / f"frame_{i:02d}.png"
        if p.exists():
            from PIL import Image
            img = Image.open(p)
            images_np.append(np.array(img.convert("RGB")))
    if len(images_np) < 2:
        print("ERROR: Need at least 2 Re10k images.")
        sys.exit(1)
    print(f"\nLoaded {len(images_np)} images, shape={images_np[0].shape}")

    # ---- Stage 1: Run VGGT (get geometry supervision) ----
    print("\n--- VGGT: Geometry Teacher ---")
    t0 = time.perf_counter()
    from src.background import VGGT, VGGTStub
    from src.background.vggt_wrapper import _opencv_rdf_to_yup_points

    is_real_vggt = VGGT is not VGGTStub
    if not is_real_vggt:
        print("ERROR: Need real VGGT.")
        sys.exit(1)

    vggt = VGGT(max_resolution=512, estimate_ground=True, estimate_drivable=True,
                known_camera_height_m=1.5)
    vggt.build()
    vggt_result = vggt.infer(images_np[:2])  # Use 2 views
    vggt_time = time.perf_counter() - t0

    # VGGT pointmap is already in Y-up (wrapper converts internally).
    pointmap_yup = vggt_result["pointmap"][0].reshape(-1, 3)  # (H*W, 3) first frame
    scale = vggt_result.get("scale_factor", 1.0)
    print(f"  VGGT: {len(pointmap_yup)} points, scale ×{scale:.3f}, {vggt_time:.1f}s")
    print(f"  Pointmap range X: [{pointmap_yup[:,0].min():.2f}, {pointmap_yup[:,0].max():.2f}]")
    print(f"  Pointmap range Y: [{pointmap_yup[:,1].min():.2f}, {pointmap_yup[:,1].max():.2f}]")
    print(f"  Pointmap range Z: [{pointmap_yup[:,2].min():.2f}, {pointmap_yup[:,2].max():.2f}]")

    # ---- Stage 2: Run MVSplat (get Gaussians with rendering opacity) ----
    print("\n--- MVSplat: Gaussian Generator ---")
    t0 = time.perf_counter()

    from src.foreground import MVSplatWrapper

    fg = MVSplatWrapper()
    checkpoint_path = str(_MVSPLAT_ROOT / "checkpoints" / "re10k.ckpt")
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    fg.build(checkpoint_path=checkpoint_path, mvsplat_root=str(_MVSPLAT_ROOT))

    # Prepare MVSplat inputs (2 views, 256×256).
    v = 2
    H, W = 256, 256
    from PIL import Image as PILImage
    resized = []
    for img in images_np[:v]:
        im_pil = PILImage.fromarray(img).resize((W, H), PILImage.LANCZOS)
        resized.append(np.array(im_pil, dtype=np.float32) / 255.0)
    mv_imgs_np = np.stack(resized, axis=0)
    mv_imgs = torch.from_numpy(mv_imgs_np).permute(0, 3, 1, 2).unsqueeze(0).float().cuda()

    # Use VGGT OpenCV-world poses for MVSplat (coordinate consistency).
    wfc_ocv = vggt_result.get("camera_poses_wfc_ocv")
    if wfc_ocv is not None and len(wfc_ocv) >= v:
        poses_np = wfc_ocv[:v]
        poses = torch.from_numpy(poses_np).float().unsqueeze(0).cuda()
        print("  Using VGGT OpenCV poses for MVSplat.")
    else:
        print("ERROR: VGGT did not return camera_poses_wfc_ocv.")
        sys.exit(1)

    K = torch.tensor(
        [[[300, 0, W / 2], [0, 300, H / 2], [0, 0, 1]]],
        device="cuda"
    ).repeat(1, v, 1, 1).float()

    fg_result = fg.infer(mv_imgs, poses, K)
    g_data = fg_result["gaussians"]
    mvsplat_time = time.perf_counter() - t0

    # Convert MVSplat means: OpenCV RDF → Y-up.
    means_yup = _opencv_rdf_to_yup_points(g_data.means.copy())
    print(f"  MVSplat: {len(means_yup)} Gaussians, {mvsplat_time:.1f}s")
    print(f"  Opacity: mean={g_data.opacities.mean():.4f}, pass_rate(>0.5)="
          f"{(g_data.opacities > 0.5).mean():.4f}")
    print(f"  Means Y-up range X: [{means_yup[:,0].min():.2f}, {means_yup[:,0].max():.2f}]")
    print(f"  Means Y-up range Y: [{means_yup[:,1].min():.2f}, {means_yup[:,1].max():.2f}]")
    print(f"  Means Y-up range Z: [{means_yup[:,2].min():.2f}, {means_yup[:,2].max():.2f}]")

    # ---- Stage 3: Assign VGGT depth-based occupancy labels to each Gaussian ----
    print("\n--- Per-Gaussian Occupancy Labeling (VGGT depth projection) ---")
    t0 = time.perf_counter()

    # Strategy: for each MVSplat Gaussian, project to VGGT camera views,
    # compare depth vs VGGT depth → label as occupied/free/unknown.

    # Get VGGT pointmap in camera space (before Y-up conversion) for depth.
    # The raw pointmap (OpenCV RDF, unit-scale) gives us per-pixel 3D in world coords.
    # We need depth = Z in camera frame. Use the VGGT W2C pose to transform.

    # VGGT outputs in its infer(): pointmap_world (Y-up, scaled), camera_poses_wfc_ocv (W2C).
    # We need: raw pointmap (before Y-up + scale) in world frame → transform to camera → get Z.

    # Actually, let's use a simpler approach:
    # Use the VGGT pointmap in Y-up (same frame as MVSplat means) with KD-tree,
    # but with a much larger threshold and a free-space model.

    # NEW APPROACH: VGGT depth ray labeling
    # 1. Get VGGT depth per pixel (Z in camera frame)
    # 2. For each Gaussian, project to VGGT camera view
    # 3. Compare Gaussian depth vs VGGT depth

    # Step 1: Get VGGT camera parameters for the first view
    wfc_ocv = vggt_result["camera_poses_wfc_ocv"]  # (V, 4, 4) W2C OpenCV
    if wfc_ocv is None or len(wfc_ocv) < 1:
        print("ERROR: No VGGT camera poses available")
        sys.exit(1)

    # Use first VGGT view for labeling
    w2c_ocv = wfc_ocv[0]  # (4, 4) world→camera, OpenCV
    c2w_ocv = np.linalg.inv(w2c_ocv)  # camera→world

    # VGGT intrinsics (from VGGT wrapper's default or estimate from pointmap resolution)
    # VGGT pointmap is at resolution (H_pm, W_pm) = (280, 504) for resolution 512
    # Use approximate intrinsics matching the input image aspect ratio
    H_vggt, W_vggt = 280, 504
    h_img, w_img = images_np[0].shape[:2]  # (720, 1280)
    # Scale intrinsics: VGGT processes at ~512px, pointmap at lower res
    fx = 300 * (W_vggt / 256)  # ~591
    fy = 300 * (H_vggt / 256)  # ~328
    cx, cy = W_vggt / 2, H_vggt / 2
    K_vggt = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    # Step 2: Compute VGGT depth map (Z in camera frame for each pointmap pixel)
    # Get raw pointmap in OpenCV world frame (before Y-up conversion)
    # The VGGT wrapper internally converts to Y-up; we need the original OpenCV coords.
    # Workaround: convert Y-up pointmap BACK to OpenCV using inverse of _opencv_rdf_to_yup_points
    # R_yup_to_ocv = diag(1, -1, -1)^{-1} = diag(1, -1, -1)  (it's its own inverse)
    R_yup_to_ocv = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
    pointmap_ocv = (R_yup_to_ocv @ pointmap_yup.T).T  # (N, 3) in OpenCV world frame

    # Transform pointmap to camera frame: P_cam = R_w2c @ P_world + t_w2c
    R_w2c = w2c_ocv[:3, :3]
    t_w2c = w2c_ocv[:3, 3]
    pointmap_cam = (R_w2c @ pointmap_ocv.T + t_w2c.reshape(3, 1)).T  # (N, 3) in camera frame

    # Depth = Z coordinate in OpenCV camera (forward)
    vggt_depths_cam = pointmap_cam[:, 2]  # (N,) — OpenCV RDF: Z=forward

    # Build per-pixel VGGT depth map
    # Pointmap pixels are at regular grid positions
    pm_pixels_y = np.arange(H_vggt)  # rows
    pm_pixels_x = np.arange(W_vggt)  # cols
    # Pointmap is (H, W, 3) → pixel (u,v) maps to depth at that pixel
    # Reshape pointmap_cam to (H_vggt, W_vggt, 3) and take Z channel
    vggt_depth_map = pointmap_cam[:, 2].reshape(H_vggt, W_vggt)  # (280, 504)

    print(f"  VGGT depth map: {vggt_depth_map.shape}, "
          f"range=[{vggt_depth_map.min():.2f}, {vggt_depth_map.max():.2f}]")
    print(f"  VGGT depth map: {vggt_depth_map.shape}, "
          f"range=[{vggt_depth_map.min():.2f}, {vggt_depth_map.max():.2f}]")

    # Step 3: Project each MVSplat Gaussian to VGGT camera, compare depths
    # Convert MVSplat means (Y-up) → OpenCV world → VGGT camera
    means_ocv = (R_yup_to_ocv @ means_yup.T).T  # (N, 3) in OpenCV world
    means_cam = (R_w2c @ means_ocv.T + t_w2c.reshape(3, 1)).T  # (N, 3) in VGGT camera
    gauss_z_cam = means_cam[:, 2]  # depth of each Gaussian in camera frame

    # Project to VGGT depth map pixels
    gauss_u = fx * means_cam[:, 0] / np.clip(means_cam[:, 2], 1e-4, None) + cx
    gauss_v = fy * means_cam[:, 1] / np.clip(means_cam[:, 2], 1e-4, None) + cy
    gauss_u_int = np.clip(gauss_u.astype(int), 0, W_vggt - 1)
    gauss_v_int = np.clip(gauss_v.astype(int), 0, H_vggt - 1)

    # Look up VGGT depth at each Gaussian's projected pixel
    vggt_depth_at_gauss = vggt_depth_map[gauss_v_int, gauss_u_int]  # (N,)

    # Depth comparison: Gaussian depth vs VGGT depth
    delta_z = gauss_z_cam - vggt_depth_at_gauss  # >0 = behind VGGT surface, <0 = in front

    # Labeling (in camera Z space):
    #   delta_z > +threshold  → BEHIND surface → UNKNOWN (occluded)
    #   |delta_z| <= threshold → AT surface → OCCUPIED
    #   delta_z < -threshold  → IN FRONT of surface → FREE
    DEPTH_THRESH = 0.3  # meters tolerance for "at surface"

    occ_mask = np.abs(delta_z) <= DEPTH_THRESH
    free_mask = delta_z < -DEPTH_THRESH
    unknown_mask = delta_z > DEPTH_THRESH

    n_occ = int(occ_mask.sum())
    n_free = int(free_mask.sum())
    n_unknown = int(unknown_mask.sum())
    n_out_of_bounds = int((gauss_z_cam <= 0).sum())

    print(f"  Depth threshold: ±{DEPTH_THRESH}m")
    print(f"  Occupied: {n_occ} ({100*n_occ/len(means_yup):.1f}%)")
    print(f"  Free:     {n_free} ({100*n_free/len(means_yup):.1f}%)")
    print(f"  Unknown:  {n_unknown} ({100*n_unknown/len(means_yup):.1f}%)")
    print(f"  Behind cam: {n_out_of_bounds}")

    # Create training labels: occupied=1, free=0 (skip unknown and behind-cam)
    # For the binary classifier, we only use occupied and free Gaussians
    trainable_mask = occ_mask | free_mask
    depth_labels = np.where(occ_mask, 1.0, 0.0).astype(np.float32)  # occupied=1, free=0

    n_trainable = int(trainable_mask.sum())
    print(f"  Trainable (occ+free): {n_trainable} ({100*n_trainable/len(means_yup):.1f}%)")
    label_time = time.perf_counter() - t0
    print(f"  Labeling time: {label_time:.1f}s")

    # ---- Stage 4: Train Occupancy Head MLP ----
    print("\n--- Training Occupancy Head MLP ---")
    t0 = time.perf_counter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Features: means(3) + scales(3) + opacity(1) = 7
    features_np = np.column_stack([
        means_yup,
        g_data.scales,
        g_data.opacities.reshape(-1, 1),
    ]).astype(np.float32)

    # Only use trainable Gaussians (occupied + free, excluding unknown/behind-camera)
    train_idx_all = np.where(trainable_mask)[0]
    n_trainable_total = len(train_idx_all)
    if n_trainable_total < 100:
        print(f"ERROR: Only {n_trainable_total} trainable Gaussians. Cannot train MLP.")
        print("This confirms the fundamental problem: MVSplat Gaussians are not in VGGT's coordinate frame.")
        sys.exit(1)

    # Train/val split (80/20) on trainable subset.
    idx_sub = np.random.RandomState(42).permutation(n_trainable_total)
    n_train = int(n_trainable_total * 0.8)
    train_idx = train_idx_all[idx_sub[:n_train]]
    val_idx = train_idx_all[idx_sub[n_train:]]

    X_train = torch.from_numpy(features_np[train_idx]).float().to(device)
    y_train = torch.from_numpy(depth_labels[train_idx]).float().to(device)
    X_val = torch.from_numpy(features_np[val_idx]).float().to(device)
    y_val = torch.from_numpy(depth_labels[val_idx]).float().to(device)

    # Normalize features.
    x_mean = X_train.mean(dim=0)
    x_std = X_train.std(dim=0).clamp(min=1e-6)
    X_train_n = (X_train - x_mean) / x_std
    X_val_n = (X_val - x_mean) / x_std

    # Train.
    model = OccupancyHead(in_dim=7, hidden=16).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    n_epochs = 500
    train_losses = []
    val_losses = []

    # Class weight: up-weight occupied (minority class).
    n_pos = y_train.sum().item()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)

    for ep in range(n_epochs):
        model.train()
        opt.zero_grad()
        pred = model(X_train_n)
        loss = F.binary_cross_entropy(pred, y_train, weight=pos_weight.expand_as(pred))
        loss.backward()
        opt.step()
        train_losses.append(loss.item())

        if ep % 100 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_n)
                val_loss = F.binary_cross_entropy(val_pred, y_val)
                val_losses.append(val_loss.item())
                val_acc = ((val_pred > 0.5).float() == y_val).float().mean().item()
            print(f"  epoch {ep:4d}/{n_epochs}: train_loss={loss.item():.4f}, "
                  f"val_loss={val_loss.item():.4f}, val_acc={val_acc:.4f}")

    train_time = time.perf_counter() - t0

    # Final metrics.
    model.eval()
    with torch.no_grad():
        # Predict occupancy for ALL Gaussians.
        X_all = torch.from_numpy(features_np).float().to(device)
        X_all_n = (X_all - x_mean.to(device)) / x_std.to(device)
        mlp_occ_all = model(X_all_n).cpu().numpy()

        # Validation accuracy.
        val_pred_final = model(X_val_n)
        val_acc_final = ((val_pred_final > 0.5).float() == y_val).float().mean().item()
        val_precision = ((val_pred_final > 0.5) & (y_val == 1)).float().sum().item() / \
                        max((val_pred_final > 0.5).float().sum().item(), 1)

    n_total = len(means_yup)
    n_mlp_occ = int((mlp_occ_all > 0.5).sum())
    print(f"  Training complete: {n_epochs} epochs, {train_time:.1f}s")
    print(f"  Val accuracy: {val_acc_final:.4f}, precision: {val_precision:.4f}")
    print(f"  MLP predicts occupied (>0.5): {n_mlp_occ}/{n_total} "
          f"({100*n_mlp_occ/n_total:.1f}%)")
    print(f"  MLP occupancy: mean={mlp_occ_all.mean():.4f}, "
          f"median={np.median(mlp_occ_all):.4f}")

    # ---- Stage 5: BEV Comparison ----
    print("\n--- BEV Comparison: Opacity vs VGGT-Label vs MLP-Predicted ---")
    t0 = time.perf_counter()

    # Compute unified bounds from all data.
    all_x = np.concatenate([means_yup[:, 0], pointmap_yup[:, 0]])
    all_z = np.concatenate([means_yup[:, 2], pointmap_yup[:, 2]])
    x_margin = (all_x.max() - all_x.min()) * 0.1 + 0.5
    z_margin = (all_z.max() - all_z.min()) * 0.1 + 0.5
    x_range = (float(all_x.min() - x_margin), float(all_x.max() + x_margin))
    z_range = (float(all_z.min() - z_margin), float(all_z.max() + z_margin))

    # (a) Baseline: opacity-based BEV
    bev_opacity = project_to_bev(
        means_yup, g_data.opacities, g_data.scales,
        x_range=x_range, z_range=z_range,
    )

    # (b) Oracle: VGGT depth-label-based BEV (occupied=1, free=0, unknown=0)
    vggt_depth_labels_bev = np.where(unknown_mask, 0.0, depth_labels)  # treat unknown as free for BEV
    bev_vggt_labels = project_to_bev(
        means_yup, vggt_depth_labels_bev, g_data.scales,
        x_range=x_range, z_range=z_range,
    )

    # (c) MLP: predicted occupancy BEV
    bev_mlp = project_to_bev(
        means_yup, mlp_occ_all, g_data.scales,
        x_range=x_range, z_range=z_range,
    )

    proj_time = time.perf_counter() - t0

    # ---- Metrics ----
    metrics = {
        "experiment": "occupancy_head_poc",
        "num_gaussians": int(len(means_yup)),
        "vggt_points": int(len(pointmap_yup)),
        "scale_factor": float(scale),
        "depth_threshold_m": DEPTH_THRESH,
        "vggt_label_occupied_pct": round(100 * n_occ / len(means_yup), 2),
        "vggt_label_free_pct": round(100 * n_free / len(means_yup), 2),
        "vggt_label_unknown_pct": round(100 * n_unknown / len(means_yup), 2),
        "vggt_label_trainable_pct": round(100 * n_trainable / len(means_yup), 2),
        "opacity_mean": round(float(g_data.opacities.mean()), 4),
        "opacity_pass_rate_0_5": round(float((g_data.opacities > 0.5).mean()), 4),
        "mlp_occ_mean": round(float(mlp_occ_all.mean()), 4),
        "mlp_occ_pass_rate_0_5": round(float((mlp_occ_all > 0.5).mean()), 4),
        "mlp_val_accuracy": round(val_acc_final, 4),
        "mlp_val_precision": round(val_precision, 4),
        "mlp_train_epochs": n_epochs,
        "mlp_train_time_s": round(train_time, 1),
        "vggt_time_s": round(vggt_time, 1),
        "mvsplat_time_s": round(mvsplat_time, 1),
        "label_time_s": round(label_time, 1),
        "proj_time_s": round(proj_time, 3),
        "gpu_memory_mb": round(
            torch.cuda.max_memory_allocated() / 1024**2, 1
        ) if torch.cuda.is_available() else 0,
    }

    # BEV metrics for each variant.
    for name, bev in [("opacity", bev_opacity), ("vggt_label", bev_vggt_labels), ("mlp", bev_mlp)]:
        m = compute_bev_metrics(bev, f"bev_{name}")
        metrics.update(m)

    # Key comparison.
    metrics["coverage_improvement_mlp_vs_opacity"] = round(
        metrics["bev_mlp_coverage_t0_3"] - metrics["bev_opacity_coverage_t0_3"], 6
    )
    metrics["coverage_improvement_vggt_vs_opacity"] = round(
        metrics["bev_vggt_label_coverage_t0_3"] - metrics["bev_opacity_coverage_t0_3"], 6
    )

    # ---- Save ----
    metrics_path = _OUTPUT_DIR / "occupancy_head_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[Metrics] Saved: {metrics_path}")

    # ---- Visualization ----
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    variants = [
        ("Opacity (baseline)", bev_opacity, "hot"),
        ("VGGT Labels (oracle)", bev_vggt_labels, "hot"),
        ("MLP Occupancy (trained)", bev_mlp, "hot"),
    ]

    for col, (title, bev, cmap) in enumerate(variants):
        ax_main = axes[0, col]
        im = ax_main.imshow(bev, cmap=cmap, origin="lower", vmin=0, vmax=1)
        ax_main.set_title(title, fontsize=11)
        ax_main.set_xlabel("X (cells)")
        ax_main.set_ylabel("Z (cells)")
        plt.colorbar(im, ax=ax_main, shrink=0.8)

        ax_bin = axes[1, col]
        ax_bin.imshow((bev > 0.3).astype(float), cmap="Greys", origin="lower")
        ax_bin.set_title(f"{title} (thresh=0.3)", fontsize=10)
        n_occ = (bev > 0.3).sum()
        ax_bin.text(0.5, -0.15, f"Occupied cells: {n_occ}",
                    transform=ax_bin.transAxes, ha="center", fontsize=9)

    # Loss curve.
    ax_loss = axes[0, 3]
    ax_loss.plot(train_losses, alpha=0.3, color="blue", linewidth=0.5, label="Train")
    # Smooth.
    if len(train_losses) > 50:
        kernel = np.ones(20) / 20
        smoothed = np.convolve(train_losses, kernel, mode="valid")
        ax_loss.plot(np.arange(len(smoothed)) + 10, smoothed, color="blue", linewidth=1.5)
    ax_loss.set_title("Train Loss", fontsize=11)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("BCE Loss")
    ax_loss.grid(True, alpha=0.3)

    # Summary text.
    ax_text = axes[1, 3]
    ax_text.axis("off")
    summary_lines = [
        "=== Occupancy Head POC ===",
        f"Gaussians: {len(means_yup)}",
        f"VGGT points: {len(pointmap_yup)}",
        f"Depth threshold: ±{DEPTH_THRESH}m",
        f"Trainable: {n_trainable}/{len(means_yup)} ({100*n_trainable/len(means_yup):.1f}%)",
        "",
        "--- VGGT Depth Labels ---",
        f"Occupied: {n_occ} ({100*n_occ/len(means_yup):.1f}%)",
        f"Free: {n_free} ({100*n_free/len(means_yup):.1f}%)",
        f"Unknown: {n_unknown} ({100*n_unknown/len(means_yup):.1f}%)",
        "",
        "--- Opacity (baseline) ---",
        f"Mean: {metrics['opacity_mean']:.4f}",
        f"Pass rate (>0.5): {metrics['opacity_pass_rate_0_5']:.4f}",
        f"BEV cov(>0.3): {metrics['bev_opacity_coverage_t0_3']:.6f}",
        "",
        "--- VGGT Labels (oracle) ---",
        f"BEV cov(>0.3): {metrics['bev_vggt_label_coverage_t0_3']:.6f}",
        "",
        "--- MLP (trained) ---",
        f"Mean occ: {metrics['mlp_occ_mean']:.4f}",
        f"Pass rate (>0.5): {metrics['mlp_occ_pass_rate_0_5']:.4f}",
        f"BEV cov(>0.3): {metrics['bev_mlp_coverage_t0_3']:.6f}",
        f"Val acc: {val_acc_final:.4f}",
        f"Val precision: {val_precision:.4f}",
        "",
        "--- Improvements ---",
        f"VGGT vs opacity: {metrics['coverage_improvement_vggt_vs_opacity']:.6f}",
        f"MLP vs opacity: {metrics['coverage_improvement_mlp_vs_opacity']:.6f}",
    ]
    ax_text.text(0.05, 0.95, "\n".join(summary_lines),
                 transform=ax_text.transAxes, fontsize=8.5,
                 verticalalignment="top", fontfamily="monospace")

    plt.tight_layout()
    vis_path = _OUTPUT_DIR / "occupancy_head_comparison.png"
    plt.savefig(vis_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved: {vis_path}")

    # ---- Print Summary ----
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<40} {'Opacity':>12} {'VGGT':>12} {'MLP':>12}")
    print("-" * 76)
    for key_display, op_key, vggt_key, mlp_key in [
        ("Coverage (t=0.3)", "bev_opacity_coverage_t0_3", "bev_vggt_label_coverage_t0_3", "bev_mlp_coverage_t0_3"),
        ("Coverage (t=0.1)", "bev_opacity_coverage_t0_1", "bev_vggt_label_coverage_t0_1", "bev_mlp_coverage_t0_1"),
        ("Density", "bev_opacity_density", "bev_vggt_label_density", "bev_mlp_density"),
        ("Occupied cells (>0.3)", "bev_opacity_occupied_cells", "bev_vggt_label_occupied_cells", "bev_mlp_occupied_cells"),
        ("Max value", "bev_opacity_max", "bev_vggt_label_max", "bev_mlp_max"),
    ]:
        print(f"{key_display:<40} {metrics[op_key]:>12.6f} {metrics[vggt_key]:>12.6f} {metrics[mlp_key]:>12.6f}")

    print(f"\nCoverage improvement (VGGT over opacity): {metrics['coverage_improvement_vggt_vs_opacity']:.6f}")
    print(f"Coverage improvement (MLP over opacity):  {metrics['coverage_improvement_mlp_vs_opacity']:.6f}")
    print("\nDone.")


if __name__ == "__main__":
    main()

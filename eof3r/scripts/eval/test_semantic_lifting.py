#!/usr/bin/env python3
"""Test 2D→3D semantic lifting via Gaussian Grouping-style identity encoding.

Pipeline:
  1. YOLO+SAM2 → 2D masks + class labels (from existing wrapper)
  2. MVSplat → feedforward Gaussians (from existing wrapper)
  3. MVSplatWrapper.train_identity_encoding() → per-Gaussian object IDs
  4. Verify: cluster identity encodings → object-level Gaussian groups

Output: outputs/eval/semantic_lifting/

Usage:
  python eof3r/scripts/eval/test_semantic_lifting.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_CODE_ROOT = _REPO_ROOT / "eof3r"
sys.path.insert(0, str(_CODE_ROOT))

MVSPLAT_ROOT = _REPO_ROOT / "baselines" / "mvsplat"
PUBLIC_DIR = _REPO_ROOT / "data" / "public" / "re10k_samples"
OUTPUT_DIR = _REPO_ROOT / "outputs" / "eval" / "semantic_lifting"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    import torch
    if not torch.cuda.is_available():
        print("ERROR: CUDA required for this test.")
        return

    print("=" * 70)
    print("2D→3D Semantic Lifting Test")
    print("=" * 70)

    # ---- 1. Load real image ----
    from PIL import Image
    fp = PUBLIC_DIR / "frame_00.png"
    if not fp.exists():
        print(f"ERROR: no test image at {fp}")
        return
    image = np.array(Image.open(fp).convert("RGB"))
    print(f"\n[1] Image: {image.shape}")

    # ---- 2. YOLO+SAM2 segmentation ----
    print("\n[2] YOLO+SAM2 segmentation...")
    t0 = time.perf_counter()
    from src.segmentation import SAM2, SAM2Stub
    is_real = SAM2 is not SAM2Stub
    seg = SAM2(model_size="base", min_mask_area=500, use_yolo=is_real)
    if is_real:
        seg.build()
    seg_result = seg.detect_and_segment(image, yolo_conf=0.35)
    print(f"  Objects: {len(seg_result['labels'])} — {seg_result['labels']}")
    print(f"  Time: {time.perf_counter() - t0:.1f}s")

    # Build 2D mask: 0=background, 1..K=objects.
    masks_2d = seg_result["masks"]  # (K, H, W) bool
    H_img, W_img = image.shape[:2]
    mask_2d = np.zeros((H_img, W_img), dtype=np.int32)
    for i in range(len(masks_2d)):
        mask_2d[masks_2d[i]] = i + 1  # object class 1, 2, ...
    num_classes = len(masks_2d) + 1  # +1 for background
    print(f"  2D mask classes (incl bg): {num_classes}")

    # ---- 3. VGGT background + camera pose ----
    print("\n[3] VGGT background (for camera pose)...")
    t0 = time.perf_counter()
    from src.background import VGGT, VGGTStub
    is_real_vggt = VGGT is not VGGTStub
    bg = VGGT(max_resolution=512, estimate_ground=True, estimate_drivable=False,
              known_camera_height_m=1.5)  # enable scale recovery
    if is_real_vggt:
        bg.build()
    bg_result = bg.infer([image, image])  # 2 frames
    # Use world-from-camera in Y-up for the first frame.
    wfc_yup = bg_result["camera_poses"][0]  # (4, 4)
    print(f"  Pose: cam_pos=({wfc_yup[0,3]:.2f}, {wfc_yup[1,3]:.2f}, {wfc_yup[2,3]:.2f})")
    print(f"  Time: {time.perf_counter() - t0:.1f}s")

    # ---- 4. MVSplat feedforward Gaussians ----
    print("\n[4] MVSplat feedforward Gaussians...")
    t0 = time.perf_counter()
    from src.foreground import MVSplatWrapper
    fg = MVSplatWrapper()
    fg.build(checkpoint_path=str(MVSPLAT_ROOT / "checkpoints" / "re10k.ckpt"),
             mvsplat_root=str(MVSPLAT_ROOT))

    # Use two copies of the image (MVSplat needs ≥2 views).
    H, W = 256, 256
    im_pil = Image.fromarray(image).resize((W, H), Image.LANCZOS)
    im_arr = np.array(im_pil, dtype=np.float32) / 255.0
    mv = torch.from_numpy(np.stack([im_arr, im_arr])).permute(0, 3, 1, 2).unsqueeze(0).float().cuda()

    # Get VGGT's OpenCV-world-from-camera poses for MVSplat.
    wfc_ocv = bg_result.get("camera_poses_wfc_ocv")
    if wfc_ocv is not None and len(wfc_ocv) >= 2:
        poses = torch.from_numpy(wfc_ocv[:2]).float().unsqueeze(0).cuda()
    else:
        poses = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1).float().cuda()
        poses[0, 1, 0, 3] = 0.3

    K = torch.tensor([[[300, 0, W/2], [0, 300, H/2], [0, 0, 1]]], device="cuda").repeat(1, 2, 1, 1).float()
    fg_result = fg.infer(mv, poses, K)
    g = fg_result["gaussians"]

    # Convert to Y-up.
    from src.background.vggt_wrapper import _opencv_rdf_to_yup_points
    g = type("GaussianData", (), {
        "means": _opencv_rdf_to_yup_points(g.means),
        "opacities": g.opacities,
        "scales": g.scales,
        "covariances": g.covariances,
        "harmonics": g.harmonics,
        "rotations": g.rotations,
        "identity_encoding": None,
    })()
    print(f"  Gaussians: {len(g.means)}")
    print(f"  Time: {time.perf_counter() - t0:.1f}s")

    # ---- 5. 3D Semantic Lifting via VGGT pointmap bridge ----
    print(f"\n[5] 3D Semantic Lifting (VGGT pointmap → MVSplat Gaussians)...")
    t0 = time.perf_counter()

    # Strategy: Use VGGT pointmap as the 2D→3D bridge.
    # 1. For each pixel in the 2D mask, get its 3D position from VGGT pointmap.
    # 2. Build a 3D labeled point cloud from VGGT.
    # 3. For each MVSplat Gaussian, find nearest VGGT point → inherit its label.
    from scipy.spatial import KDTree

    vggt_pm = bg_result["pointmap"][0]  # (H_pm, W_pm, 3) in Y-up
    pm_h, pm_w = vggt_pm.shape[:2]

    # Resize 2D mask to VGGT pointmap resolution.
    mask_resized = np.zeros((pm_h, pm_w), dtype=np.int32)
    for i in range(pm_h):
        for j in range(pm_w):
            src_i = int(i * H_img / pm_h)
            src_j = int(j * W_img / pm_w)
            mask_resized[i, j] = mask_2d[min(src_i, H_img-1), min(src_j, W_img-1)]

    # Build TWO KD-trees: one for foreground VGGT points (labeled by masks),
    # one for ALL VGGT points.  Assign each Gaussian to the nearest class.
    ds = 4
    vggt_pm_ds = vggt_pm[::ds, ::ds]
    mask_ds = mask_resized[::ds, ::ds]
    vggt_flat = vggt_pm_ds.reshape(-1, 3)
    vggt_labels_flat = mask_ds.ravel()

    # Per-class KD-trees — prevents majority class (couch) from dominating.
    unique_fg = sorted(set(vggt_labels_flat[vggt_labels_flat > 0]))
    class_trees: dict[int, KDTree] = {}
    class_points: dict[int, np.ndarray] = {}
    for cls_id in unique_fg:
        cls_mask = vggt_labels_flat == cls_id
        pts = vggt_flat[cls_mask]
        if len(pts) >= 5:
            class_trees[cls_id] = KDTree(pts)
            class_points[cls_id] = pts
    print(f"  Classes with ≥5 VGGT points: {list(class_trees.keys())}")

    if not class_trees:
        print("  ERROR: No class has enough VGGT points.")
        per_gaussian_labels = np.zeros(len(g.means), dtype=np.int32)
        return

    # Build background KD-tree (all VGGT points).
    tree_bg = KDTree(vggt_flat)

    # For each Gaussian: find distance to closest point of EACH class + background.
    per_gaussian_labels = np.zeros(len(g.means), dtype=np.int32)
    n_assigned = 0

    # Compute per-class intra-class median distance for adaptive thresholding.
    for cls_id, tree in class_trees.items():
        cls_pts = class_points[cls_id]
        # Intra-class typical distance (self-query for K=2).
        if len(cls_pts) >= 3:
            d_self, _ = tree.query(cls_pts, k=2)
            intra_dist = float(np.median(d_self[:, 1]))
        else:
            intra_dist = 5.0  # fallback

        # Query MVSplat Gaussians against this class + background.
        d_cls, _ = tree.query(g.means, k=1)
        d_bg, _ = tree_bg.query(g.means, k=1)

        # Adaptive threshold: max(intra_class_dist * 3, 3.0m absolute min).
        thresh = max(intra_dist * 3.0, 3.0)

        # Assign to this class if:
        #   d_cls < adaptive_threshold  (close enough to this class's points)
        #   d_cls < 2.0 * d_bg          (significantly closer to FG than BG)
        cls_assign = (d_cls < thresh) & (d_cls < 2.0 * d_bg)
        n_new = int(cls_assign.sum())
        new_mask = cls_assign & (per_gaussian_labels == 0)
        per_gaussian_labels[new_mask] = cls_id
        n_assigned += int(new_mask.sum())

        cls_name = seg_result["labels"][cls_id - 1] if cls_id <= len(seg_result["labels"]) else f"cls_{cls_id}"
        print(f"  Class {cls_id} ({cls_name:12s}): intra_dist={intra_dist:.2f}m, thresh={thresh:.2f}m, "
              f"{n_new} candidates → {int(new_mask.sum())} assigned")

    print(f"  Total FG-assigned Gaussians: {n_assigned}/{len(g.means)} ({100*n_assigned/len(g.means):.1f}%)")

    train_time = time.perf_counter() - t0
    print(f"  Lifting time: {train_time:.1f}s")

    # ---- 6. Analyze per-Gaussian semantics ----
    print("\n[6] Per-Gaussian semantic analysis:")
    unique_labels, counts = np.unique(per_gaussian_labels, return_counts=True)
    for lbl, cnt in sorted(zip(unique_labels, counts), key=lambda x: -x[1]):
        if lbl == 0:
            name = "background"
        elif lbl <= len(seg_result["labels"]):
            name = seg_result["labels"][lbl - 1]
        else:
            name = f"obj_{lbl}"
        pct = 100 * cnt / len(per_gaussian_labels)
        print(f"  Class {lbl} ({name:12s}): {cnt:6d} Gaussians ({pct:5.1f}%)")

    # Per-object Gaussian groups (excluding background).
    obj_ids = [l for l in unique_labels if l > 0]
    print(f"\n  Object-level Gaussian groups: {len(obj_ids)}")
    for obj_id in sorted(obj_ids):
        obj_mask = per_gaussian_labels == obj_id
        obj_means = g.means[obj_mask]
        if obj_id <= len(seg_result["labels"]):
            name = seg_result["labels"][obj_id - 1]
        else:
            name = f"obj_{obj_id}"
        print(f"    {name:12s}: {obj_mask.sum():6d} Gaussians, "
              f"center=({obj_means[:,0].mean():.2f}, {obj_means[:,1].mean():.2f}, {obj_means[:,2].mean():.2f}), "
              f"extent_xz=({obj_means[:,0].max()-obj_means[:,0].min():.2f}, {obj_means[:,2].max()-obj_means[:,2].min():.2f})")

    # ---- 7. Object-level BEV footprints ----
    print("\n[7] Object-level BEV footprints (per-Gaussian group):")
    from src.fusion import BEVProjector
    proj = BEVProjector({"bev_range": "auto", "bev_target_cells": 400, "height_filter": [-1.0, 8.0]})
    for obj_id in sorted(obj_ids):
        obj_mask = per_gaussian_labels == obj_id
        if obj_mask.sum() < 10:
            continue
        obj_means = g.means[obj_mask]
        obj_ops = g.opacities[obj_mask]
        obj_scales = g.scales[obj_mask]
        name = seg_result["labels"][obj_id - 1] if obj_id <= len(seg_result["labels"]) else f"obj_{obj_id}"
        try:
            obj_bev = proj.project_gaussians_to_bev(
                means=obj_means, opacities=obj_ops, scales=obj_scales,
            )
            occupied = (obj_bev > 0.3).sum()
            print(f"    {name:12s}: BEV {obj_bev.shape}, occupied cells = {occupied}")
        except Exception as e:
            print(f"    {name:12s}: BEV projection failed: {e}")

    # ---- 8. Save results ----
    results = {
        "num_gaussians": int(len(g.means)),
        "num_classes": int(num_classes),
        "num_objects_detected": int(len(obj_ids)),
        "per_class_gaussian_count": {str(k): int(v) for k, v in zip(unique_labels, counts)},
        "object_labels": seg_result["labels"],
        "lift_time_s": round(train_time, 1),
        "method": "VGGT_pointmap_bridge + KDTree_NN",
    }
    with open(OUTPUT_DIR / "semantic_lifting.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults: {OUTPUT_DIR / 'semantic_lifting.json'}")
    print("Semantic lifting test complete!")


if __name__ == "__main__":
    main()

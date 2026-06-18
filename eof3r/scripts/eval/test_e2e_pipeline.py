#!/usr/bin/env python3
"""End-to-end pipeline test with quantitative metrics.

PIPELINE:
  segmentation (SAM2 stub) → foreground (MVSplat) → background (VGGT stub)
  → fusion (BEV projection) → costmap (Nav2 format)

Stages that use real pretrained models:
  - foreground: MVSplat (requires GPU, re10k checkpoint)

Stages using stubs (blocked by GitHub TLS):
  - segmentation: SAM2Stub (synthetic masks)
  - background: VGGTStub (synthetic pointmap)

All metrics are saved to outputs/eval/e2e_metrics.json.

Usage:
  python scripts/eval/test_e2e_pipeline.py [--skip-mvsplat] [--output-dir outputs/eval]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add repo root to path.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_CODE_ROOT = _REPO_ROOT / "eof3r"
sys.path.insert(0, str(_CODE_ROOT))

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import warnings

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = str(_REPO_ROOT / "outputs" / "eval")
MVSPLAT_ROOT = _REPO_ROOT / "baselines" / "mvsplat"


def get_config() -> dict:
    """Load config from configs/default.yaml."""
    import yaml
    config_path = _CODE_ROOT / "configs" / "default.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def _load_real_image(public_dir: Path) -> np.ndarray | None:
    """Load the first real test image from data/public/ if available."""
    frame_path = public_dir / "frame_00.png"
    if not frame_path.exists():
        return None
    from PIL import Image
    img = Image.open(frame_path)
    return np.array(img.convert("RGB"))


def _make_default_poses(v: int, device: str = "cpu") -> torch.Tensor:
    """Create default identity-like C2W poses (OpenCV convention).

    Camera i is offset by 0.3*i in X, looking along +Z.
    """
    poses_list = []
    for vi in range(v):
        pose = torch.eye(4, device=device)
        pose[0, 3] = 0.3 * vi  # X offset between views
        poses_list.append(pose)
    return torch.stack(poses_list).unsqueeze(0)  # (1, V, 4, 4)


def _load_real_images(public_dir: Path, n: int = 4) -> list[np.ndarray] | None:
    """Load multiple real test images."""
    images = []
    for i in range(n):
        frame_path = public_dir / f"frame_{i:02d}.png"
        if not frame_path.exists():
            return None
        from PIL import Image
        img = Image.open(frame_path)
        images.append(np.array(img.convert("RGB")))
    return images


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def compute_bev_metrics(bev: np.ndarray, name: str = "bev") -> dict:
    """Compute BEV occupancy metrics.

    Args:
        bev: (H, W) float32 occupancy grid [0, 1].
        name: Label prefix for metric keys.

    Returns:
        Dict of metrics.
    """
    thresholds = [0.1, 0.3, 0.5, 0.7]
    total_cells = bev.size
    metrics: dict = {}

    for t in thresholds:
        coverage = float((bev > t).sum()) / max(total_cells, 1)
        # Normalize key: be_v_occupancy_coverage_t0_1 etc.
        key = f"{name}_occupancy_coverage_t{t}".replace(".", "_")
        metrics[key] = round(coverage, 6)

    # Density: average occupancy over occupied cells.
    occupied_mask = bev > 0.01
    density = float(bev[occupied_mask].mean()) if occupied_mask.sum() > 0 else 0.0
    metrics[f"{name}_occupancy_density"] = round(density, 6)

    # Spatial extent (approximate area in m²).
    # Assumes 0.05 m/cell resolution as default.
    res = 0.05
    extent_area = float(occupied_mask.sum()) * res * res
    metrics[f"{name}_spatial_extent_m2"] = round(extent_area, 4)

    return metrics


def compute_gaussian_metrics(means: np.ndarray, opacities: np.ndarray, alpha_threshold: float = 0.5) -> dict:
    """Compute Gaussian quality metrics.

    Args:
        means: (N, 3) Gaussian centers in Y-up.
        opacities: (N,) opacity values.
        alpha_threshold: Threshold for pass rate.

    Returns:
        Dict of metrics.
    """
    metrics: dict = {}
    metrics["num_gaussians_total"] = int(len(opacities))

    if len(opacities) > 0:
        metrics["opacity_mean"] = round(float(opacities.mean()), 6)
        metrics["opacity_std"] = round(float(opacities.std()), 6)
        metrics["opacity_min"] = round(float(opacities.min()), 6)
        metrics["opacity_max"] = round(float(opacities.max()), 6)

        pass_rate = float((opacities >= alpha_threshold).sum()) / len(opacities)
        metrics["alpha_threshold_pass_rate"] = round(pass_rate, 6)

        # Spatial range.
        for axis, label in enumerate(["x", "y", "z"]):
            metrics[f"gaussian_spatial_range_{label}_min"] = round(float(means[:, axis].min()), 4)
            metrics[f"gaussian_spatial_range_{label}_max"] = round(float(means[:, axis].max()), 4)
            metrics[f"gaussian_spatial_range_{label}_span"] = round(
                float(means[:, axis].max() - means[:, axis].min()), 4
            )
    else:
        metrics.update({
            "opacity_mean": 0, "opacity_std": 0, "opacity_min": 0, "opacity_max": 0,
            "alpha_threshold_pass_rate": 0,
            "gaussian_spatial_range_x_min": 0, "gaussian_spatial_range_x_max": 0,
            "gaussian_spatial_range_x_span": 0,
            "gaussian_spatial_range_y_min": 0, "gaussian_spatial_range_y_max": 0,
            "gaussian_spatial_range_y_span": 0,
            "gaussian_spatial_range_z_min": 0, "gaussian_spatial_range_z_max": 0,
            "gaussian_spatial_range_z_span": 0,
        })

    return metrics


def compute_fusion_metrics(
    fg_bev: np.ndarray,
    bg_bev: np.ndarray,
    bg_drivable: np.ndarray,
) -> dict:
    """Compute fusion consistency metrics.

    Args:
        fg_bev: (H, W) foreground BEV occupancy [0, 1].
        bg_bev: (H, W) background BEV occupancy [0, 1].
        bg_drivable: (H, W) drivable mask (True=drivable, False=obstacle).

    Returns:
        Dict of metrics.
    """
    total = fg_bev.size
    t = 0.3

    fg_occ = fg_bev > t
    bg_occ = bg_bev > t
    drivable = bg_drivable > 0.5

    metrics: dict = {}

    # Coverage.
    metrics["fg_bev_coverage"] = round(float(fg_occ.sum()) / max(total, 1), 6)
    metrics["bg_bev_coverage"] = round(float(bg_occ.sum()) / max(total, 1), 6)

    # Overlap IoU between fg and bg.
    intersection = float((fg_occ & bg_occ).sum())
    union = float((fg_occ | bg_occ).sum())
    metrics["fg_bg_overlap_iou"] = round(intersection / max(union, 1), 6)

    # Drivable-occupancy conflict: fg=occupied but bg=says-drivable.
    conflict = fg_occ & drivable
    metrics["drivable_occupancy_conflict_rate"] = round(
        float(conflict.sum()) / max(fg_occ.sum(), 1), 6
    )

    return metrics


def compute_costmap_metrics(costmap: np.ndarray) -> dict:
    """Compute costmap validity metrics.

    Args:
        costmap: (H, W) uint8 costmap (0=free, 254=lethal, 255=unknown).

    Returns:
        Dict of metrics.
    """
    total = costmap.size
    metrics: dict = {}
    metrics["costmap_min"] = int(costmap.min())
    metrics["costmap_max"] = int(costmap.max())
    metrics["lethal_cell_count"] = int((costmap >= 254).sum())
    metrics["free_cell_count"] = int((costmap == 0).sum())
    metrics["costmap_completeness"] = round(
        float((costmap != 255).sum()) / max(total, 1), 6
    )
    return metrics


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_visualizations(
    bev_occupancy: np.ndarray,
    costmap: np.ndarray,
    output_dir: Path,
    stage_times: dict,
    all_metrics: dict,
) -> None:
    """Save BEV and costmap visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. BEV occupancy (foreground)
    im0 = axes[0, 0].imshow(bev_occupancy, cmap="hot", origin="lower", vmin=0, vmax=1)
    axes[0, 0].set_title("BEV Occupancy (Foreground)")
    axes[0, 0].set_xlabel("X (cells)")
    axes[0, 0].set_ylabel("Y (cells)")
    plt.colorbar(im0, ax=axes[0, 0], shrink=0.8)

    # 2. BEV binary (threshold 0.3)
    axes[0, 1].imshow(
        (bev_occupancy > 0.3).astype(float), cmap="Greys", origin="lower"
    )
    axes[0, 1].set_title("BEV Footprint (threshold=0.3)")

    # 3. Costmap
    im2 = axes[0, 2].imshow(costmap, cmap="gray_r", origin="lower", vmin=0, vmax=254)
    axes[0, 2].set_title("Nav2 Costmap (0=free, 254=lethal)")
    plt.colorbar(im2, ax=axes[0, 2], shrink=0.8)

    # 4. Coverage vs threshold
    thresholds = np.linspace(0.0, 1.0, 50)
    coverages = [(bev_occupancy > t).sum() / bev_occupancy.size for t in thresholds]
    axes[1, 0].plot(thresholds, coverages, "b-", linewidth=2)
    axes[1, 0].set_title("Occupancy Coverage vs Threshold")
    axes[1, 0].set_xlabel("Threshold")
    axes[1, 0].set_ylabel("Coverage")
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Stage timing bar chart
    stages = list(stage_times.keys())
    times_s = [stage_times[s] for s in stages]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(stages)))
    axes[1, 1].bar(stages, times_s, color=colors)
    axes[1, 1].set_title("Per-Stage Wall Time")
    axes[1, 1].set_ylabel("Time (s)")
    axes[1, 1].tick_params(axis="x", rotation=45)
    for i, (_s, t) in enumerate(zip(stages, times_s, strict=False)):
        axes[1, 1].text(i, t + 0.02, f"{t:.2f}s", ha="center", fontsize=8)

    # 6. Text summary
    axes[1, 2].axis("off")
    summary_lines = [
        f"Gaussians: {all_metrics.get('num_gaussians_total', 'N/A')}",
        f"Opacity mean: {all_metrics.get('opacity_mean', 'N/A')}",
        f"Pass rate (>0.5): {all_metrics.get('alpha_threshold_pass_rate', 'N/A')}",
        f"BEV coverage (>0.3): {all_metrics.get('bev_occupancy_coverage_t0_3', 'N/A')}",
        f"Costmap lethal: {all_metrics.get('lethal_cell_count', 'N/A')}",
        f"Costmap completeness: {all_metrics.get('costmap_completeness', 'N/A')}",
        f"Total time: {sum(times_s):.2f}s",
    ]
    axes[1, 2].text(
        0.05, 0.95, "\n".join(summary_lines),
        transform=axes[1, 2].transAxes,
        fontsize=10, verticalalignment="top",
        fontfamily="monospace",
    )

    plt.tight_layout()
    save_path = output_dir / "e2e_pipeline_visualization.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="EOF3R End-to-End Pipeline Test")
    parser.add_argument(
        "--skip-mvsplat", action="store_true",
        help="Skip MVSplat (use synthetic Gaussians instead)."
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Output directory for metrics and visualizations."
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="checkpoints/re10k.ckpt",
        help="MVSplat checkpoint path (relative to baselines/mvsplat/)."
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = get_config()

    print("=" * 70)
    print("EOF3R End-to-End Pipeline Test")
    print("=" * 70)

    stage_times: dict[str, float] = {}
    all_metrics: dict = {}
    gpu_memory_peak_mb = 0.0

    # ---- Stage 1: Segmentation ----
    print("\n" + "-" * 50)
    print("[Stage 1/5] Segmentation")
    print("-" * 50)
    t0 = time.perf_counter()

    from src.segmentation import SAM2, SAM2Stub

    # Use real SAM2 if available, else stub.
    is_real_sam2 = SAM2 is not SAM2Stub
    print(f"  Using: {'REAL SAM2Wrapper' if is_real_sam2 else 'SAM2Stub (synthetic)'}")

    seg = SAM2(
        model_size=config.get("segmentation", {}).get("model_size", "base"),
        min_mask_area=config.get("segmentation", {}).get("min_mask_area", 500),
    )
    if is_real_sam2:
        try:
            seg.build()
        except Exception as e:
            print(f"  WARNING: SAM2 build failed ({e}), falling back to stub.")
            seg = SAM2Stub(
                model_size=config.get("segmentation", {}).get("model_size", "base"),
                min_mask_area=config.get("segmentation", {}).get("min_mask_area", 500),
            )
            is_real_sam2 = False

    # Load test images from data/public/ or use synthetic fallback.
    public_dir = _REPO_ROOT / "data" / "public" / "re10k_samples"
    test_image = _load_real_image(public_dir) if public_dir.exists() else None
    if test_image is None:
        h_img, w_img = 480, 640
        test_image = np.random.randint(0, 255, (h_img, w_img, 3), dtype=np.uint8)
        print("  Using synthetic image (no public data found).")
    else:
        print(f"  Loaded real image: {test_image.shape}")

    seg_result = seg.segment(test_image, box_prompt=False)

    stage_times["segmentation"] = round(time.perf_counter() - t0, 4)
    all_metrics["seg_num_objects"] = int(len(seg_result["masks"]))
    all_metrics["seg_is_real"] = is_real_sam2
    print(f"  Objects detected: {all_metrics['seg_num_objects']}")
    print(f"  Labels: {seg_result.get('labels', 'N/A')}")
    print(f"  Time: {stage_times['segmentation']:.3f}s")

    # ---- Stage 2: Background ----
    print("\n" + "-" * 50)
    print("[Stage 2/5] Background")
    print("-" * 50)
    t0 = time.perf_counter()

    from src.background import VGGT, VGGTStub

    is_real_vggt = VGGT is not VGGTStub
    print(f"  Using: {'REAL VGGTWrapper' if is_real_vggt else 'VGGTStub (synthetic)'}")

    bg = VGGT(
        max_resolution=config.get("background", {}).get("max_resolution", 512),
        estimate_ground=config.get("background", {}).get("estimate_ground", True),
        estimate_drivable=config.get("background", {}).get("estimate_drivable", True),
    )
    if is_real_vggt:
        try:
            bg.build()
        except Exception as e:
            print(f"  WARNING: VGGT build failed ({e}), falling back to stub.")
            bg = VGGTStub(
                max_resolution=config.get("background", {}).get("max_resolution", 512),
                estimate_ground=True, estimate_drivable=True,
            )
            is_real_vggt = False

    # Provide real multi-view images if available, else duplicate.
    real_imgs = _load_real_images(public_dir, n=2)
    if real_imgs is not None:
        bg_images = real_imgs
        print(f"  Using {len(bg_images)} real Re10k views for VGGT.")
    else:
        bg_images = [test_image.copy() for _ in range(2)]
    bg_result = bg.infer(bg_images)

    stage_times["background"] = round(time.perf_counter() - t0, 4)
    all_metrics["bg_pointmap_shape"] = str(bg_result["pointmap"].shape)
    all_metrics["bg_num_views"] = len(bg_result["camera_poses"])
    all_metrics["bg_is_real"] = is_real_vggt
    print(f"  Pointmap shape: {bg_result['pointmap'].shape}")
    print(f"  Camera poses: {len(bg_result['camera_poses'])} views")
    print(f"  Time: {stage_times['background']:.3f}s")

    # ---- Stage 3: Foreground (MVSplat) ----
    print("\n" + "-" * 50)
    print("[Stage 3/5] Foreground (MVSplat)")
    print("-" * 50)
    t0 = time.perf_counter()

    used_real_mvsplat = False  # track whether real MVSplat ran (for coord conversion)

    if args.skip_mvsplat or not torch.cuda.is_available():
        print("  WARNING: Skipping MVSplat (--skip-mvsplat or no CUDA). Using synthetic Gaussians.")
        n_synthetic = 5000
        means = np.random.randn(n_synthetic, 3).astype(np.float32) * 3.0
        means[:, 1] = np.abs(means[:, 1]) * 0.8  # Y-up, mostly above ground
        opacities = np.random.rand(n_synthetic).astype(np.float32) * 0.9 + 0.1
        scales = np.random.rand(n_synthetic, 3).astype(np.float32) * 0.3 + 0.05
        covariances = np.array(
            [np.diag(s) for s in scales], dtype=np.float32
        )
        g_data = type("GaussianData", (), {
            "means": means,
            "opacities": opacities,
            "scales": scales,
            "covariances": covariances,
            "harmonics": np.zeros((n_synthetic, 3, 1), dtype=np.float32),
            "rotations": None,
        })()
        print("  Using synthetic Gaussian data.")
    else:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Loading MVSplat from: {args.checkpoint}")

        from src.foreground import MVSplatWrapper

        fg = MVSplatWrapper()
        checkpoint_path = str(MVSPLAT_ROOT / args.checkpoint)
        if not os.path.exists(checkpoint_path):
            print(f"  ERROR: Checkpoint not found: {checkpoint_path}")
            print("  Falling back to synthetic Gaussians.")
            n_synthetic = 5000
            means = np.random.randn(n_synthetic, 3).astype(np.float32) * 3.0
            means[:, 1] = np.abs(means[:, 1]) * 0.8
            opacities = np.random.rand(n_synthetic).astype(np.float32) * 0.9 + 0.1
            scales = np.random.rand(n_synthetic, 3).astype(np.float32) * 0.3 + 0.05
            covariances = np.array([np.diag(s) for s in scales], dtype=np.float32)
            g_data = type("GaussianData", (), {
                "means": means,
                "opacities": opacities,
                "scales": scales,
                "covariances": covariances,
                "harmonics": np.zeros((n_synthetic, 3, 1), dtype=np.float32),
                "rotations": None,
            })()
        else:
            fg.build(checkpoint_path=checkpoint_path, mvsplat_root=str(MVSPLAT_ROOT))

            # Use real images and VGGT-estimated poses for coordinate consistency.
            # VGGT outputs are in OpenCV convention; MVSplat also uses OpenCV natively,
            # so we feed OpenCV-world-from-camera poses → MVSplat Gaussians are in
            # the SAME OpenCV world frame as VGGT's raw pointmap.
            real_imgs = _load_real_images(public_dir, n=4) if public_dir.exists() else None
            v = 2  # Use 2 context views
            H, W = 256, 256

            if is_real_vggt and real_imgs is not None:
                # Use VGGT's estimated OpenCV-world-from-camera poses as MVSplat C2W.
                wfc_ocv = bg_result.get("camera_poses_wfc_ocv")
                if wfc_ocv is not None and len(wfc_ocv) >= v:
                    mvsplat_poses_np = wfc_ocv[:v]  # (V, 4, 4) OpenCV C2W
                    poses = torch.from_numpy(mvsplat_poses_np).float().unsqueeze(0).cuda()
                    print("  Using VGGT-estimated OpenCV poses for MVSplat (coords aligned).")
                else:
                    poses = _make_default_poses(v, device="cuda")
                    print("  WARNING: VGGT poses unavailable, using default poses.")
            else:
                poses = _make_default_poses(v, device="cuda")
                print("  WARNING: VGGT not available, using default poses.")

            if real_imgs is not None:
                # Resize real images to 256x256 for MVSplat.
                from PIL import Image as PILImage
                resized = []
                for img in real_imgs[:v]:
                    im_pil = PILImage.fromarray(img).resize((W, H), PILImage.LANCZOS)
                    im_arr = np.array(im_pil, dtype=np.float32) / 255.0  # [0, 1]
                    resized.append(im_arr)
                mv_imgs = np.stack(resized, axis=0)  # (V, H, W, 3)
                mv_imgs = torch.from_numpy(mv_imgs).permute(0, 3, 1, 2).unsqueeze(0).float().cuda()
                # (1, V, 3, H, W)
                print(f"  Using {v} real Re10k views for MVSplat.")
            else:
                mv_imgs = torch.rand(1, v, 3, H, W, device="cuda")
                print("  Using random images (no real Re10k data).")

            # Build intrinsics: use default pinhole K (MVSplat handles FOV from K).
            K = torch.tensor(
                [[[300, 0, W / 2], [0, 300, H / 2], [0, 0, 1]]],
                device="cuda"
            ).repeat(1, v, 1, 1).float()

            fg_result = fg.infer(mv_imgs, poses, K)
            g_data = fg_result["gaussians"]
            used_real_mvsplat = True

    stage_times["foreground"] = round(time.perf_counter() - t0, 4)

    # Convert MVSplat Gaussian means from OpenCV RDF → Y-up if we used VGGT coords.
    # MVSplat outputs in the world frame defined by input extrinsics.
    # Since we used VGGT's OpenCV-world poses, the Gaussians are in OpenCV RDF.
    # Convert to Y-up for fusion with VGGT pointmap (which is already Y-up).
    if used_real_mvsplat and is_real_vggt:
        from src.background.vggt_wrapper import _opencv_rdf_to_yup_points
        g_data = type("GaussianData", (), {
            "means": _opencv_rdf_to_yup_points(g_data.means),
            "opacities": g_data.opacities,
            "scales": g_data.scales,  # scale magnitudes are frame-invariant
            "covariances": g_data.covariances,  # cov needs rotation too, but keeping simple for now
            "harmonics": g_data.harmonics,
            "rotations": g_data.rotations,
        })()
        print("  Converted MVSplat Gaussians: OpenCV RDF → Y-up.")

    # Gaussian metrics.
    gauss_metrics = compute_gaussian_metrics(
        g_data.means, g_data.opacities,
        alpha_threshold=config.get("occupancy", {}).get("alpha_threshold", 0.5)
    )
    all_metrics.update(gauss_metrics)
    print(f"  Gaussians: {all_metrics['num_gaussians_total']}")
    print(f"  Opacity: mean={all_metrics['opacity_mean']:.4f}, "
          f"min={all_metrics['opacity_min']:.4f}, max={all_metrics['opacity_max']:.4f}")
    print(f"  Alpha pass rate: {all_metrics['alpha_threshold_pass_rate']:.4f}")
    print(f"  Mean X: {g_data.means[:, 0].mean():.2f} Y: {g_data.means[:, 1].mean():.2f} Z: {g_data.means[:, 2].mean():.2f}")
    print(f"  Time: {stage_times['foreground']:.3f}s")

    # GPU memory.
    if torch.cuda.is_available():
        gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
        gpu_memory_peak_mb = gpu_memory_mb
        all_metrics["peak_gpu_memory_mb"] = round(gpu_memory_mb, 2)
        print(f"  Peak GPU memory: {gpu_memory_mb:.1f} MB")

    # ---- Stage 4: Fusion (BEV projection) ----
    print("\n" + "-" * 50)
    print("[Stage 4/5] Fusion (BEV projection)")
    print("-" * 50)
    t0 = time.perf_counter()

    from src.fusion import BEVProjector

    fusion_cfg = config.get("fusion", {})
    projector = BEVProjector(fusion_cfg)

    # Project foreground Gaussians to BEV.
    fg_bev_grid = projector.project_gaussians_to_bev(
        means=g_data.means,
        opacities=g_data.opacities,
        scales=g_data.scales,
    )

    # Project background pointmap to BEV.
    # Take the first frame's pointmap, convert Y-up means to BEV.
    bg_pointmap = bg_result["pointmap"][0]  # (H, W, 3) in Y-up
    h_pm, w_pm = bg_pointmap.shape[:2]
    bg_means = bg_pointmap.reshape(-1, 3)
    # Use synthetic opacities for background.
    bg_opacities = np.random.rand(len(bg_means)).astype(np.float32) * 0.5 + 0.3
    bg_scales = np.full((len(bg_means), 3), 0.1, dtype=np.float32)
    bg_bev_grid = projector.project_gaussians_to_bev(
        means=bg_means,
        opacities=bg_opacities,
        scales=bg_scales,
    )

    # Resize drivable mask to match BEV grid using projector's helper.
    bg_drivable = bg_result["drivable_mask"]  # (h_out, w_out) bool
    h_bev, w_bev = fg_bev_grid.shape
    bg_drivable_resized = BEVProjector._resize_to_grid(
        bg_drivable.astype(np.float32), h_bev, w_bev
    ) > 0.5

    # Fuse.
    fused_bev = projector.fuse_foreground_background(
        fg_bev_grid, bg_bev_grid, bg_drivable_resized
    )

    stage_times["fusion"] = round(time.perf_counter() - t0, 4)

    # BEV metrics.
    bev_metrics = compute_bev_metrics(fg_bev_grid, "bev")
    all_metrics.update(bev_metrics)
    print(f"  BEV shape: {fg_bev_grid.shape}")
    print(f"  Occupancy coverage (t=0.3): {bev_metrics['bev_occupancy_coverage_t0_3']:.4f}")
    print(f"  Occupancy density: {bev_metrics['bev_occupancy_density']:.4f}")
    print(f"  Spatial extent: {bev_metrics['bev_spatial_extent_m2']:.2f} m²")

    # Fusion metrics.
    fus_metrics = compute_fusion_metrics(fg_bev_grid, bg_bev_grid, bg_drivable_resized)
    all_metrics.update(fus_metrics)
    print(f"  FG coverage: {fus_metrics['fg_bev_coverage']:.4f}")
    print(f"  BG coverage: {fus_metrics['bg_bev_coverage']:.4f}")
    print(f"  FG/BG overlap IoU: {fus_metrics['fg_bg_overlap_iou']:.4f}")
    print(f"  Drivable conflict rate: {fus_metrics['drivable_occupancy_conflict_rate']:.4f}")
    print(f"  Time: {stage_times['fusion']:.3f}s")

    # ---- Stage 5: Costmap ----
    print("\n" + "-" * 50)
    print("[Stage 5/5] Costmap Generation")
    print("-" * 50)
    t0 = time.perf_counter()

    from src.costmap import CostmapGenerator

    costmap_cfg = config.get("costmap", {})
    cg = CostmapGenerator(costmap_cfg)
    costmap_grid, cost_metrics_obj = cg.generate_costmap(fused_bev, bev_semantic=None)

    stage_times["costmap"] = round(time.perf_counter() - t0, 4)

    cost_metrics = compute_costmap_metrics(costmap_grid)
    # Also merge in the CostmapMetrics fields.
    cost_metrics["costmap_wall_time_ms"] = round(cost_metrics_obj.wall_time_ms, 2)
    all_metrics.update(cost_metrics)
    print(f"  Costmap shape: {costmap_grid.shape}")
    print(f"  Min/Max: {cost_metrics['costmap_min']}/{cost_metrics['costmap_max']}")
    print(f"  Free cells: {cost_metrics['free_cell_count']}")
    print(f"  Lethal cells: {cost_metrics['lethal_cell_count']}")
    print(f"  Completeness: {cost_metrics['costmap_completeness']:.4f}")
    print(f"  Time: {stage_times['costmap']:.3f}s")

    # ---- Save results ----
    all_metrics["stage_times"] = stage_times
    all_metrics["total_wall_time_s"] = round(sum(stage_times.values()), 4)
    all_metrics["peak_gpu_memory_mb"] = round(gpu_memory_peak_mb, 2)

    # Save JSON.
    metrics_path = output_dir / "e2e_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\n[Metrics] Saved: {metrics_path}")

    # Save visualization.
    save_visualizations(
        bev_occupancy=fg_bev_grid,
        costmap=costmap_grid,
        output_dir=output_dir,
        stage_times=stage_times,
        all_metrics=all_metrics,
    )

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"{'Stage':<20} {'Time (s)':<12} {'Status'}")
    print("-" * 50)
    for stage, t in stage_times.items():
        status = "OK" if t > 0 else "SKIPPED"
        print(f"  {stage:<18} {t:<12.3f} {status}")
    print("-" * 50)
    print(f"  {'TOTAL':<18} {all_metrics['total_wall_time_s']:<12.3f}")
    print()
    print("Key Metrics:")
    def _safe_print(key, fmt="s"):
        val = all_metrics.get(key, "N/A")
        if isinstance(val, float):
            print(f"  {key:<26} {val:.4f}")
        else:
            print(f"  {key:<26} {val}")
    _safe_print("num_gaussians_total")
    _safe_print("opacity_mean")
    _safe_print("alpha_threshold_pass_rate")
    _safe_print("bev_occupancy_coverage_t0_3")
    _safe_print("bev_spatial_extent_m2")
    _safe_print("fg_bg_overlap_iou")
    _safe_print("drivable_occupancy_conflict_rate")
    _safe_print("costmap_completeness")
    _safe_print("lethal_cell_count")
    _safe_print("peak_gpu_memory_mb")
    print(f"  {'total_wall_time_s':<26} {all_metrics['total_wall_time_s']:.2f}s")
    print()
    print("ALL METRICS SAVED TO:", metrics_path)
    print("VISUALIZATION SAVED TO:", output_dir / "e2e_pipeline_visualization.png")
    print("\nPipeline test completed successfully.")


if __name__ == "__main__":
    main()

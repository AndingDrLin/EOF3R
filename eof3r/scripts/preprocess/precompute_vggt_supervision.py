"""Pre-compute VGGT supervision for Phase B training.

This script runs in the eof3r conda env (Python 3.10 + VGGT) to generate
per-Gaussian supervision labels that the ReSplat training loop reads from disk.

For each Re10k scene:
1. Load images + camera poses
2. Run VGGT inference → depth maps, point maps, refined poses
3. For each context view pair:
   a. Run ReSplat-style Gaussian generation (or use pre-computed Gaussians)
   b. Project Gaussians to VGGT cameras
   c. Label each Gaussian as occupied/free/unknown
   d. Extract surface point cloud for Chamfer loss
4. Save to disk as .pt files

Output structure:
    {output_dir}/{train|test}/{scene_id}/
        images.pt          -- (V, 3, H, W) input images
        depth.pt           -- (V, H, W) VGGT depth maps
        poses.pt           -- (V, 4, 4) camera poses (world-from-camera)
        intrinsics.pt      -- (V, 3, 3) camera intrinsics
        surface_points.pt  -- (M, 3) VGGT surface point cloud
        labels.pt          -- dict with occupied/free/unknown masks

Usage:
    # Pre-compute for all Re10k scenes
    python eof3r/scripts/preprocess/precompute_vggt_supervision.py

    # Pre-compute for specific split
    python eof3r/scripts/preprocess/precompute_vggt_supervision.py --split train

    # Pre-compute with custom config
    python eof3r/scripts/preprocess/precompute_vggt_supervision.py --config eof3r/configs/phase_b.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from eof3r.src.training.supervision import (
    label_gaussians_by_vggt_projection,
    compute_vggt_surface_points,
    merge_multi_view_labels,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_vggt_wrapper(config: dict):
    """Load VGGT wrapper with configured parameters."""
    from eof3r.src.background.vggt_wrapper import VGGTWrapper

    vggt = VGGTWrapper(
        max_resolution=config.get("max_resolution", 512),
        estimate_ground=True,
        estimate_drivable=True,
    )
    vggt.build()
    return vggt


def load_re10k_scenes(
    data_root: Path,
    split: str,
    max_scenes: int | None = None,
) -> list[dict]:
    """Load Re10k scene list.

    Args:
        data_root: Path to Re10k data directory.
        split: "train" or "test".
        max_scenes: Maximum number of scenes to process.

    Returns:
        List of scene dicts with keys: scene_id, images, cameras.
    """
    # Re10k data is stored as .torch chunk files
    chunk_dir = data_root / split
    if not chunk_dir.exists():
        raise FileNotFoundError(
            f"Re10k data not found at {chunk_dir}\n"
            f"Expected structure: {data_root}/{{train,test}}/*.torch"
        )

    chunk_files = sorted(chunk_dir.glob("*.torch"))
    if not chunk_files:
        raise FileNotFoundError(f"No .torch files found in {chunk_dir}")

    scenes = []
    for chunk_file in chunk_files:
        chunk_data = torch.load(chunk_file, weights_only=False)
        for scene in chunk_data:
            scenes.append({
                "scene_id": scene["key"],
                "cameras": scene["cameras"],  # (N, 18)
                "images": scene["images"],    # list of JPEG bytes
            })
            if max_scenes and len(scenes) >= max_scenes:
                return scenes

    return scenes


def decode_re10k_image(image_data, target_size: tuple[int, int] = (256, 256)):
    """Decode Re10k image to numpy array.

    Handles both JPEG bytes and raw torch tensor formats.
    """
    from PIL import Image
    import io

    # Re10k stores images as torch tensors of JPEG-encoded bytes
    if isinstance(image_data, torch.Tensor):
        image_bytes = bytes(image_data.numpy().tolist())
    elif isinstance(image_data, bytes):
        image_bytes = image_data
    else:
        raise TypeError(f"Unexpected image type: {type(image_data)}")

    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize(target_size, Image.BILINEAR)
    return np.array(img)


def extract_re10k_cameras(
    cameras_tensor: torch.Tensor,
    image_size: tuple[int, int] = (256, 256),
):
    """Extract intrinsics and extrinsics from Re10k camera format.

    Re10k format: (N, 18) where:
        - [:4] = intrinsics (fx, fy, cx, cy) normalized by image size
        - [4:6] = unused
        - [6:18] = flattened 3x4 world-to-camera matrix

    Args:
        cameras_tensor: (N, 18) camera parameters.
        image_size: (H, W) target image size for denormalizing intrinsics.

    Returns:
        intrinsics: (N, 3, 3) with intrinsics scaled to pixel coordinates.
        extrinsics: (N, 4, 4) camera-from-world.
    """
    N = cameras_tensor.shape[0]
    H, W = image_size

    # Intrinsics (normalized → pixel)
    K = torch.eye(3).unsqueeze(0).repeat(N, 1, 1)
    K[:, 0, 0] = cameras_tensor[:, 0] * W  # fx * W
    K[:, 1, 1] = cameras_tensor[:, 1] * H  # fy * H
    K[:, 0, 2] = cameras_tensor[:, 2] * W  # cx * W
    K[:, 1, 2] = cameras_tensor[:, 3] * H  # cy * H

    # Extrinsics (3x4 → 4x4)
    w2c = torch.zeros(N, 4, 4)
    w2c[:, :3, :4] = cameras_tensor[:, 6:18].reshape(N, 3, 4)
    w2c[:, 3, 3] = 1.0

    # Invert to get camera-from-world (c2w)
    c2w = torch.inverse(w2c)

    return K, c2w


def process_scene(
    scene: dict,
    vggt_wrapper,
    output_dir: Path,
    config: dict,
    device: str = "cuda",
) -> bool:
    """Process a single scene: run VGGT, label Gaussians, save supervision.

    Args:
        scene: Scene dict with scene_id, images, cameras.
        vggt_wrapper: Initialized VGGT wrapper.
        output_dir: Output directory for this scene.
        config: Processing configuration.
        device: Device for computation.

    Returns:
        True if successful, False if skipped (e.g., too few views).
    """
    scene_id = scene["scene_id"]
    images_raw = scene["images"]
    cameras_raw = scene["cameras"]

    # Limit views
    max_views = config.get("max_views", 2)
    if len(images_raw) < max_views:
        logger.debug(f"Skipping {scene_id}: only {len(images_raw)} views")
        return False

    # Select views (take first max_views for now; could sample)
    images_raw = images_raw[:max_views]
    cameras_raw = cameras_raw[:max_views]

    # Decode images
    target_size = config.get("image_shape", (256, 256))
    images_np = [decode_re10k_image(img, target_size) for img in images_raw]

    # Extract cameras (with image size for denormalizing intrinsics)
    K, c2w = extract_re10k_cameras(cameras_raw, image_size=target_size)

    # Run VGGT inference
    try:
        vggt_output = vggt_wrapper.infer(images_np)
    except Exception as e:
        logger.warning(f"VGGT failed on {scene_id}: {e}")
        return False

    # Extract VGGT outputs
    # pointmap is in Y-up world coords (V, H', W', 3)
    # camera_poses_wfc_ocv is world-from-camera in OpenCV (V, 4, 4)
    vggt_pointmap = torch.from_numpy(vggt_output["pointmap"]).float()  # (V, H', W', 3)
    vggt_wfc_ocv = torch.from_numpy(vggt_output["camera_poses_wfc_ocv"]).float()  # (V, 4, 4)

    # Compute depth maps: project world points into camera frame → Z is depth
    # depth_v(u,v) = (R_cw @ point_w + t_cw)_z
    V, H_pm, W_pm, _ = vggt_pointmap.shape
    vggt_depth = torch.zeros(V, H_pm, W_pm)
    for v in range(V):
        wfc = vggt_wfc_ocv[v]  # (4, 4) world-from-camera
        cfw = torch.inverse(wfc)  # (4, 4) camera-from-world
        R = cfw[:3, :3]  # (3, 3)
        t = cfw[:3, 3]   # (3,)
        # Reshape pointmap for batch matmul: (H*W, 3)
        pts = vggt_pointmap[v].reshape(-1, 3)  # (H*W, 3)
        pts_cam = (R @ pts.T).T + t  # (H*W, 3) in camera frame
        depth_v = pts_cam[:, 2].reshape(H_pm, W_pm).abs()  # |Z|-depth (positive)
        vggt_depth[v] = depth_v

    # For each view, compute surface points
    all_surface_points = []
    for v in range(max_views):
        depth_v = vggt_depth[v]  # (H, W)
        K_v = K[v]  # (3, 3)
        pose_v = vggt_wfc_ocv[v]  # (4, 4)

        points = compute_vggt_surface_points(
            depth_v, K_v, pose_v,
            subsample=config.get("point_subsample", 4),
        )
        all_surface_points.append(points)

    # Merge surface points from all views
    surface_points = torch.cat(all_surface_points, dim=0)  # (M_total, 3)

    # Normalize scene to unit scale for stable training
    # The VGGT scale recovery can produce very large coordinates (100s of meters).
    # Normalizing to ~1 unit makes Chamfer loss and learning rates consistent.
    centroid = surface_points.mean(dim=0)  # (3,)
    scale = surface_points.std(dim=0).mean().clamp(min=1e-6)  # scalar
    # Normalize: shift to origin, scale to unit variance
    surface_points = (surface_points - centroid) / scale
    vggt_pointmap_norm = (vggt_pointmap - centroid) / scale
    # Update depth maps to normalized coordinates
    for v in range(max_views):
        wfc = vggt_wfc_ocv[v]
        cfw = torch.inverse(wfc)
        R = cfw[:3, :3]
        t = cfw[:3, 3]
        pts = vggt_pointmap_norm[v].reshape(-1, 3)
        pts_cam = (R @ pts.T).T + t
        vggt_depth[v] = pts_cam[:, 2].abs().reshape(H_pm, W_pm)
    # Normalize poses (translate camera positions)
    for v in range(max_views):
        vggt_wfc_ocv[v, :3, 3] = (vggt_wfc_ocv[v, :3, 3] - centroid) / scale

    # Save images
    images_tensor = torch.stack([
        torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        for img in images_np
    ])  # (V, 3, H, W)

    # Save depth maps (normalized)
    depth_tensor = vggt_depth  # (V, H, W)

    # Save poses (world-from-camera, normalized)
    poses_tensor = vggt_wfc_ocv  # (V, 4, 4)

    # Save intrinsics
    intrinsics_tensor = K  # (V, 3, 3)

    # Create output directory
    scene_dir = output_dir / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    # Save tensors
    torch.save(images_tensor, scene_dir / "images.pt")
    torch.save(depth_tensor, scene_dir / "depth.pt")
    torch.save(poses_tensor, scene_dir / "poses.pt")
    torch.save(intrinsics_tensor, scene_dir / "intrinsics.pt")
    torch.save(surface_points, scene_dir / "surface_points.pt")

    # Note: per-Gaussian labels are computed online during training
    # (they depend on the current Gaussian positions, which change every step)
    # We save the VGGT supervision that enables online labeling.

    # Save metadata
    metadata = {
        "scene_id": scene_id,
        "num_views": max_views,
        "image_shape": list(target_size),
        "num_surface_points": surface_points.shape[0],
        "vggt_scale_factor": float(vggt_output.get("scale_factor", 1.0)),
        "norm_centroid": centroid.tolist(),
        "norm_scale": float(scale),
    }
    with open(scene_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute VGGT supervision for Phase B training"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Path to Re10k data directory (default: $EOF3R_DATA/raw/re10k)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: $EOF3R_DATA/processed/vggt_supervision)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split to process",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Maximum number of scenes to process (for debugging)",
    )
    parser.add_argument(
        "--max-views",
        type=int,
        default=2,
        help="Number of context views per scene",
    )
    parser.add_argument(
        "--image-shape",
        type=int,
        nargs=2,
        default=[256, 256],
        help="Target image size (H W)",
    )
    parser.add_argument(
        "--point-subsample",
        type=int,
        default=4,
        help="Subsample factor for surface point cloud",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for VGGT inference",
    )
    args = parser.parse_args()

    # Resolve paths
    data_root = Path(args.data_root) if args.data_root else Path(
        os.environ.get("EOF3R_DATA", "/data/EOF3R")
    ) / "raw" / "re10k"

    output_dir = Path(args.output_dir) if args.output_dir else Path(
        os.environ.get("EOF3R_DATA", "/data/EOF3R")
    ) / "processed" / "vggt_supervision" / args.split

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("VGGT Supervision Pre-computation")
    logger.info(f"Data root: {data_root}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Max scenes: {args.max_scenes or 'all'}")
    logger.info("=" * 60)

    # Load VGGT
    logger.info("Loading VGGT...")
    vggt = load_vggt_wrapper({"max_resolution": 512})
    logger.info("VGGT loaded.")

    # Load Re10k scenes
    logger.info(f"Loading Re10k {args.split} split...")
    scenes = load_re10k_scenes(data_root, args.split, args.max_scenes)
    logger.info(f"Found {len(scenes)} scenes.")

    # Process scenes
    config = {
        "max_views": args.max_views,
        "image_shape": tuple(args.image_shape),
        "point_subsample": args.point_subsample,
    }

    success_count = 0
    fail_count = 0
    manifest_lines = []

    start_time = time.time()

    for i, scene in enumerate(scenes):
        scene_id = scene["scene_id"]

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(scenes) - i - 1) / rate
            logger.info(
                f"Processing scene {i+1}/{len(scenes)} "
                f"({rate:.2f} scenes/s, ETA: {eta:.0f}s)"
            )

        try:
            success = process_scene(scene, vggt, output_dir, config, args.device)
            if success:
                success_count += 1
                manifest_lines.append(scene_id)
            else:
                fail_count += 1
        except Exception as e:
            logger.error(f"Failed to process {scene_id}: {e}")
            fail_count += 1

    # Write manifest
    manifest_path = output_dir / "manifest.txt"
    with open(manifest_path, "w") as f:
        for line in manifest_lines:
            f.write(line + "\n")

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Done! Processed {success_count} scenes in {elapsed:.1f}s")
    logger.info(f"Failed: {fail_count}")
    logger.info(f"Manifest: {manifest_path}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

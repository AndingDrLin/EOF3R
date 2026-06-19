"""VGGT-Omega wrapper for Phase B geometry teacher.

VGGT-Omega (CVPR 2026 Oral) is the successor to VGGT with:
    - +26% depth accuracy (δ1.25=93.5% vs 67.5%)
    - 1.6× faster inference
    - Better multi-view consistency

This wrapper follows the same API as VGGTWrapper but uses VGGT-Omega.

Usage:
    vggt_omega = VGGTOmegaWrapper(max_resolution=512)
    vggt_omega.build(checkpoint="path/to/vggt_omega_1b_512.pt")
    output = vggt_omega.infer(images)
    # output: {pointmap, camera_poses, depth, depth_conf, ...}
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class VGGTOmegaWrapper:
    """Wrapper for VGGT-Omega geometry teacher.

    Follows the same API as VGGTWrapper:
        build() → infer() → {pointmap, camera_poses, depth, ...}

    The wrapper handles:
        1. Path isolation (VGGT-Omega has its own package)
        2. Checkpoint loading
        3. Output format conversion (compatible with VGGTWrapper)
    """

    def __init__(
        self,
        max_resolution: int = 512,
        estimate_ground: bool = True,
        estimate_drivable: bool = True,
    ):
        self.max_resolution = max_resolution
        self.estimate_ground = estimate_ground
        self.estimate_drivable = estimate_drivable
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._vggt_omega_root = None

    def build(
        self,
        checkpoint: Optional[str] = None,
        vggt_omega_root: Optional[str] = None,
    ) -> None:
        """Load VGGT-Omega model.

        Args:
            checkpoint: Path to VGGT-Omega checkpoint (.pt file).
                If None, looks for default location.
            vggt_omega_root: Path to VGGT-Omega repo.
        """
        # Resolve VGGT-Omega root
        if vggt_omega_root:
            self._vggt_omega_root = Path(vggt_omega_root)
        else:
            self._vggt_omega_root = Path(__file__).parents[3] / "baselines" / "vggt-omega"

        if not self._vggt_omega_root.exists():
            raise FileNotFoundError(
                f"VGGT-Omega not found at {self._vggt_omega_root}\n"
                f"Clone: git clone https://github.com/facebookresearch/vggt-omega baselines/vggt-omega"
            )

        # Add to path
        if str(self._vggt_omega_root) not in sys.path:
            sys.path.insert(0, str(self._vggt_omega_root))

        # Resolve checkpoint
        if checkpoint is None:
            # Look for default locations
            candidates = [
                self._vggt_omega_root / "checkpoints" / "vggt_omega_1b_512.pt",
                Path("outputs/checkpoints/vggt-omega/vggt_omega_1b_512.pt"),
                Path.home() / ".cache" / "huggingface" / "vggt_omega_1b_512.pt",
            ]
            for c in candidates:
                if c.exists():
                    checkpoint = str(c)
                    break

        if checkpoint is None:
            raise FileNotFoundError(
                "VGGT-Omega checkpoint not found.\n"
                "Download from: https://huggingface.co/facebook/VGGT-Omega\n"
                "Or pass checkpoint path explicitly."
            )

        # Load model
        from vggt_omega.models import VGGTOmega

        self.model = VGGTOmega().to(self.device).eval()
        state_dict = torch.load(checkpoint, map_location="cpu")
        self.model.load_state_dict(state_dict)

        logger.info(f"VGGT-Omega loaded from {checkpoint}")

    @torch.no_grad()
    def infer(self, images: list[np.ndarray]) -> dict[str, Any]:
        """Run VGGT-Omega inference on input images.

        Args:
            images: List of N uint8 RGB images (H, W, 3).

        Returns:
            Dict compatible with VGGTWrapper output:
                - pointmap: (N, H', W', 3) world points
                - camera_poses: (N, 4, 4) world-from-camera
                - camera_poses_wfc_ocv: (N, 4, 4) for MVSplat/ReSplat
                - depth: (N, H', W') depth maps
                - depth_conf: (N, H', W') depth confidence
                - scale_factor: float
                - ground_plane: (4,) if estimate_ground
        """
        if self.model is None:
            raise RuntimeError("Call build() before infer()")

        from vggt_omega.utils.load_fn import load_and_preprocess_images
        from vggt_omega.utils.pose_enc import encoding_to_camera

        # Preprocess images
        # VGGT-Omega expects image paths or PIL images
        # Convert numpy to format expected by VGGT-Omega
        from PIL import Image
        pil_images = [Image.fromarray(img) for img in images]

        # Save temporarily for VGGT-Omega loader
        import tempfile
        import os

        temp_dir = tempfile.mkdtemp()
        temp_paths = []
        for i, img in enumerate(pil_images):
            path = os.path.join(temp_dir, f"img_{i}.png")
            img.save(path)
            temp_paths.append(path)

        try:
            # Load and preprocess
            input_images = load_and_preprocess_images(
                temp_paths,
                image_resolution=self.max_resolution,
            ).to(self.device)

            # Run inference
            predictions = self.model(input_images)

            # Extract outputs
            extrinsics, intrinsics = encoding_to_camera(
                predictions["pose_enc"],
                predictions["images"].shape[-2:],
            )

            depth = predictions["depth"]  # (N, H, W)
            depth_conf = predictions["depth_conf"]  # (N, H, W)

            # Convert to numpy
            extrinsics_np = extrinsics.cpu().numpy()
            intrinsics_np = intrinsics.cpu().numpy()
            depth_np = depth.cpu().numpy()
            depth_conf_np = depth_conf.cpu().numpy()

            # Compute pointmaps from depth
            N, H, W = depth_np.shape
            pointmaps = np.zeros((N, H, W, 3))

            for v in range(N):
                K = intrinsics_np[v]
                T_w_c = extrinsics_np[v]

                # Create pixel grid
                u, v_coords = np.meshgrid(np.arange(W), np.arange(H))
                z = depth_np[v]

                # Back-project to camera frame
                x_cam = (u - K[0, 2]) * z / K[0, 0]
                y_cam = (v_coords - K[1, 2]) * z / K[1, 1]
                points_cam = np.stack([x_cam, y_cam, z], axis=-1)  # (H, W, 3)

                # Transform to world frame
                R_w_c = T_w_c[:3, :3]
                t_w_c = T_w_c[:3, 3]
                points_world = (R_w_c @ points_cam.reshape(-1, 3).T).T + t_w_c
                pointmaps[v] = points_world.reshape(H, W, 3)

            # Estimate ground plane (simple version)
            ground_plane = None
            if self.estimate_ground:
                # Use lowest 10% of points to fit plane
                all_points = pointmaps.reshape(-1, 3)
                z_values = all_points[:, 1]  # Y-up convention
                threshold = np.percentile(z_values, 10)
                ground_points = all_points[z_values < threshold]

                if len(ground_points) > 100:
                    # SVD fit
                    centroid = ground_points.mean(axis=0)
                    _, _, Vt = np.linalg.svd(ground_points - centroid)
                    normal = Vt[-1]
                    d = -np.dot(normal, centroid)
                    ground_plane = np.array([normal[0], normal[1], normal[2], d])
                else:
                    ground_plane = np.array([0, 1, 0, 0])  # Default Y-up

            # Scale factor (VGGT-Omega outputs metric scale directly)
            scale_factor = 1.0

            output = {
                "pointmap": pointmaps,
                "camera_poses": extrinsics_np,
                "camera_poses_wfc_ocv": extrinsics_np,
                "depth": depth_np,
                "depth_conf": depth_conf_np,
                "intrinsics": intrinsics_np,
                "scale_factor": scale_factor,
            }

            if ground_plane is not None:
                output["ground_plane"] = ground_plane

            return output

        finally:
            # Clean up temp files
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def to_vggt_format(self, output: dict) -> dict:
        """Convert VGGT-Omega output to VGGTWrapper format.

        This ensures compatibility with existing code that expects
        VGGTWrapper output format.
        """
        # VGGT-Omega already outputs in compatible format
        return output

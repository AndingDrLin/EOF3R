"""VGGT background estimation wrapper — real model interface.

Replaces VGGTStub when vggt is installed.

To use:
  1. Install VGGT: pip install -e baselines/vggt/ (needs torch 2.3.1+)
  2. First run downloads the checkpoint from HuggingFace automatically.
  3. If VGGT is not installed or CUDA is unavailable, falls back to VGGTStub.

API: build() loads the model; infer(images) returns pointmap, camera_poses,
ground_plane, drivable_mask in Y-up convention.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_VGGT_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "baselines" / "vggt"

_VGGT_AVAILABLE = False
if str(_VGGT_ROOT) not in sys.path:
    sys.path.insert(0, str(_VGGT_ROOT))
try:
    from vggt.models.vggt import VGGT

    _VGGT_AVAILABLE = True
except ImportError:
    pass

# Default HuggingFace model IDs.
_VGGT_CHECKPOINTS = {
    "1b": "facebook/VGGT-1B",
    "1b-commercial": "facebook/VGGT-1B-Commercial",
}


class VGGTWrapper:
    """Real VGGT background geometry estimator.

    Wraps the VGGT feedforward model to produce:
      - Dense pointmap (per-pixel 3D world coords)
      - Camera poses (per-frame world-from-camera)
      - Ground plane estimate
      - Drivable region mask
    """

    def __init__(
        self,
        max_resolution: int = 512,
        estimate_ground: bool = True,
        estimate_drivable: bool = True,
    ) -> None:
        """Initialize VGGT wrapper.

        Args:
            max_resolution: Resize images so max side ≤ this (px).
            estimate_ground: Fit a ground plane from the pointmap.
            estimate_drivable: Heuristic drivable mask from low-height points.
        """
        self._max_resolution = max_resolution
        self._estimate_ground = estimate_ground
        self._estimate_drivable = estimate_drivable
        self._model: object | None = None

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------
    def build(
        self,
        checkpoint_path: Optional[str] = None,
        model_variant: str = "1b",
    ) -> None:
        """Load VGGT model from HuggingFace Hub.

        Args:
            checkpoint_path: Path to local checkpoint, or None to auto-download.
            model_variant: "1b" or "1b-commercial" (HuggingFace repo).
        """
        if not _VGGT_AVAILABLE:
            raise ImportError(
                "VGGT is not installed. Run: pip install -e baselines/vggt/\n"
                "Requires torch>=2.3.1. Current torch: "
                f"{_get_torch_version()}"
            )
        if not _cuda_available():
            raise RuntimeError("VGGT requires CUDA. No GPU detected.")

        hf_repo = _VGGT_CHECKPOINTS.get(model_variant, model_variant)

        logger.info("Loading VGGT: %s", hf_repo)
        if checkpoint_path is not None:
            self._model = VGGT.from_pretrained(checkpoint_path).cuda().eval()
        else:
            self._model = VGGT.from_pretrained(hf_repo).cuda().eval()
        logger.info("VGGT loaded (variant=%s).", model_variant)

    # ------------------------------------------------------------------
    # infer
    # ------------------------------------------------------------------
    def infer(self, images: list[np.ndarray]) -> dict:
        """Run VGGT inference on a sequence of images.

        Args:
            images: List of N RGB images (H, W, 3), uint8 [0, 255].

        Returns:
            Dict with keys:
              - pointmap: (N, H', W', 3) float32 — 3D world points in Y-up coords.
              - camera_poses: (N, 4, 4) float32 — world-from-camera matrices.
              - ground_plane: (4,) float32 — (a, b, c, d) for ax+by+cz+d=0 in Y-up.
              - drivable_mask: (H', W') bool — estimated drivable floor region.
        """
        if self._model is None:
            raise RuntimeError("VGGT model not loaded. Call build() first.")

        import torch

        # Resize and normalize images.
        processed = [_resize_image(im, self._max_resolution) for im in images]
        # Stack: (N, 3, H, W) float32 in [0, 1].
        tensor = torch.from_numpy(np.stack(processed, axis=0)).float()
        if tensor.max() > 1.5:
            tensor = tensor / 255.0
        tensor = tensor.cuda()

        with torch.no_grad():
            predictions = self._model(tensor)

        # Extract outputs.
        # world_points: (B=1, S, H, W, 3) → strip batch dim.
        wp = predictions["world_points"][0].cpu().numpy().astype(np.float32)
        # wp is in VGGT's coordinate frame.  VGGT uses a right-handed frame where
        # the first camera is at origin looking +Z.  This is NOT the same as our Y-up
        # convention.  We leave the conversion to the fusion stage (coord_utils.py).
        # For now, store as-is and document the frame in metadata.

        # Camera poses: extract from pose_enc.
        pose_enc = predictions["pose_enc"][0].cpu().numpy()  # (S, 9)
        camera_poses = _pose_enc_to_matrices(pose_enc)

        # Ground plane: fit to bottom-center region of pointmap.
        ground_plane = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        if self._estimate_ground:
            ground_plane = _estimate_ground_plane(wp)

        # Drivable mask: heuristic — low-height flat points.
        drivable_mask = np.zeros(wp.shape[1:3], dtype=bool)
        if self._estimate_drivable and wp.size > 0:
            drivable_mask = _estimate_drivable_region(wp, ground_plane)

        n = len(images)
        logger.info(
            "VGGT inference: %d frames, pointmap %s, ground=%s.",
            n, wp.shape, tuple(ground_plane.round(3)),
        )
        return {
            "pointmap": wp,
            "camera_poses": camera_poses,
            "ground_plane": ground_plane,
            "drivable_mask": drivable_mask,
        }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _resize_image(image: np.ndarray, max_res: int, patch_size: int = 14) -> np.ndarray:
    """Resize image so max(H, W) ≤ max_res, then round to patch_size multiples.

    VGGT uses a ViT backbone with patch_size=14.  Input dimensions must be
    divisible by 14 or the patch embedder asserts.
    """
    h, w = image.shape[:2]
    scale = min(max_res / max(h, w), 1.0)
    if scale < 1.0:
        new_h, new_w = int(h * scale), int(w * scale)
        import cv2

        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = new_h, new_w
    # Round down to nearest patch_size multiple.
    h = (h // patch_size) * patch_size
    w = (w // patch_size) * patch_size
    if h != image.shape[0] or w != image.shape[1]:
        import cv2

        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    return image.transpose(2, 0, 1)  # HWC → CHW


def _pose_enc_to_matrices(pose_enc: np.ndarray) -> np.ndarray:
    """Convert VGGT pose encoding (S, 9) → (S, 4, 4) world-from-camera matrices.

    VGGT pose_enc: first 3 = translation, last 6 = rotation (6D representation).
    """
    s = pose_enc.shape[0]
    matrices = np.tile(np.eye(4, dtype=np.float32), (s, 1, 1))
    # Set translation.
    matrices[:, :3, 3] = pose_enc[:, :3]
    # 6D rotation → 3x3 matrix.
    r6d = pose_enc[:, 3:9].reshape(s, 2, 3)
    # Gram-Schmidt orthogonalization.
    a1 = r6d[:, 0, :]
    a2 = r6d[:, 1, :]
    b1 = a1 / (np.linalg.norm(a1, axis=1, keepdims=True) + 1e-8)
    b2 = a2 - np.sum(b1 * a2, axis=1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    matrices[:, :3, 0] = b1
    matrices[:, :3, 1] = b2
    matrices[:, :3, 2] = b3
    return matrices


def _estimate_ground_plane(world_points: np.ndarray) -> np.ndarray:
    """Fit ground plane (a,b,c,d) to central-bottom region of the pointmap.

    Uses the first frame's pointmap, samples central-bottom region,
    and fits a plane via SVD.
    """
    pm = world_points[0]  # (H, W, 3) — first frame
    h, w = pm.shape[:2]
    # Central 60% width, bottom 40% height.
    cx_start, cx_end = int(w * 0.2), int(w * 0.8)
    cy_start, cy_end = int(h * 0.6), h
    region = pm[cy_start:cy_end, cx_start:cx_end].reshape(-1, 3)
    # Remove extreme outliers (Y-up: Y is height).
    y_vals = region[:, 1]
    q05, q95 = np.percentile(y_vals, [5, 95])
    inliers = region[(y_vals >= q05) & (y_vals <= q95)]
    if len(inliers) < 10:
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    # SVD fit: centroid + normal.
    centroid = inliers.mean(axis=0)
    _, _, vh = np.linalg.svd(inliers - centroid, full_matrices=False)
    normal = vh[2]  # smallest singular vector
    normal = normal / (np.linalg.norm(normal) + 1e-8)
    # Ensure normal points up (positive Y in Y-up).
    if normal[1] < 0:
        normal = -normal
    d = -np.dot(normal, centroid)
    return np.array([*normal, d], dtype=np.float32)


def _estimate_drivable_region(
    wp: np.ndarray, ground_plane: np.ndarray, height_thresh: float = 0.3
) -> np.ndarray:
    """Heuristic drivable mask: points within height_thresh of the ground plane."""
    normal = ground_plane[:3]
    d = ground_plane[3]
    # Per-pixel signed distance to ground plane.
    # VGGT world_points: [..., 0]=x, [..., 1]=y, [..., 2]=z.
    # Ground plane in VGGT frame — use first-frame points.
    pm = wp[0]
    dist = np.abs(np.dot(pm.reshape(-1, 3), normal) + d)
    dist_grid = dist.reshape(pm.shape[0], pm.shape[1])
    return dist_grid < height_thresh


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _get_torch_version() -> str:
    try:
        import torch

        return torch.__version__
    except ImportError:
        return "not installed"

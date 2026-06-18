"""VGGT background estimation wrapper — real model interface.

To use:
  1. pip install git+https://github.com/facebookresearch/vggt.git
  2. First run downloads the checkpoint from HuggingFace automatically.
  3. If VGGT is not installed or CUDA is unavailable, falls back to VGGTStub.

API: build() loads the model; infer(images) returns pointmap, camera_poses,
ground_plane, drivable_mask.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Try native import first (pip-installed), then local baselines/ for dev.
_VGGT_AVAILABLE = False
try:
    from vggt.models.vggt import VGGT

    _VGGT_AVAILABLE = True
except ImportError:
    import sys

    _VGGT_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "baselines" / "vggt"
    if _VGGT_ROOT.exists():
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
        known_camera_height_m: float | None = None,
    ) -> None:
        """Initialize VGGT wrapper.

        Args:
            max_resolution: Resize images so max side ≤ this (px).
            estimate_ground: Fit a ground plane from the pointmap.
            estimate_drivable: Heuristic drivable mask from low-height points.
            known_camera_height_m: If provided, use ground plane to recover
                real-world scale (metres).  Required for robot deployment
                where camera mounting height is known.
        """
        self._max_resolution = max_resolution
        self._estimate_ground = estimate_ground
        self._estimate_drivable = estimate_drivable
        self._known_camera_height_m = known_camera_height_m
        self._model: object | None = None

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------
    def build(
        self,
        checkpoint_path: str | None = None,
        model_variant: str = "1b",
    ) -> None:
        """Load VGGT model from HuggingFace Hub.

        Args:
            checkpoint_path: Path to local checkpoint, or None to auto-download.
            model_variant: "1b" or "1b-commercial" (HuggingFace repo).
        """
        if not _VGGT_AVAILABLE:
            raise ImportError(
                "VGGT is not installed. Run:\n"
                "  pip install git+https://github.com/facebookresearch/vggt.git\n"
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
              - pointmap: (N, H', W', 3) float32 — 3D points in Y-up coords.
              - camera_poses: (N, 4, 4) float32 — world-from-camera in Y-up coords.
              - camera_poses_cfw: (N, 4, 4) float32 — camera-from-world in OpenCV.
              - ground_plane: (4,) float32 — (a, b, c, d) for ax+by+cz+d=0 in Y-up.
              - drivable_mask: (H', W') bool — estimated drivable floor region.
              - image_size_hw: (h, w) tuple of processed image dimensions.
        """
        if self._model is None:
            raise RuntimeError("VGGT model not loaded. Call build() first.")

        import torch

        # Resize and normalize images.
        processed = [_resize_image(im, self._max_resolution) for im in images]
        # All processed images must have the same spatial size for stacking.
        # _resize_image ensures uniform size via the max_resolution constraint,
        # but if input images differ in aspect ratio, force to the smallest
        # common dimensions.
        shapes = {(p.shape[1], p.shape[2]) for p in processed}  # (H, W) per image
        if len(shapes) > 1:
            h_min = min(s[0] for s in shapes)
            w_min = min(s[1] for s in shapes)
            for i in range(len(processed)):
                if processed[i].shape[1] != h_min or processed[i].shape[2] != w_min:
                    import cv2

                    processed[i] = cv2.resize(
                        processed[i].transpose(1, 2, 0), (w_min, h_min),
                        interpolation=cv2.INTER_AREA,
                    ).transpose(2, 0, 1)
            logger.warning(
                "VGGT input images had mismatched sizes; resized to (%d, %d).", h_min, w_min
            )
        # Stack: (N, 3, H, W) float32 in [0, 1].
        tensor = torch.from_numpy(np.stack(processed, axis=0)).float()
        if tensor.max() > 1.5:
            tensor = tensor / 255.0
        tensor = tensor.cuda()

        with torch.no_grad():
            predictions = self._model(tensor)

        # Extract outputs.
        # world_points: (B=1, S, H, W, 3) → strip batch dim.
        wp_ocv = predictions["world_points"][0].cpu().numpy().astype(np.float32)
        # wp_ocv is in VGGT's OpenCV coordinate frame (RDF: Right-Down-Forward).
        # The first camera is at origin looking +Z, Y points down.
        # We convert to the project's Y-up convention (Right-Up-Backward).

        # Camera poses: decode VGGT's 9D pose_enc → camera-from-world (OpenCV).
        pose_enc = predictions["pose_enc"][0].cpu().numpy()  # (S, 9)
        cfw_ocv = _pose_enc_to_matrices(pose_enc)  # camera-from-world in OpenCV

        # Convert pointmap from OpenCV RDF to Y-up (project convention).
        wp_yup = _opencv_rdf_to_yup_points(wp_ocv)

        # Convert camera poses: camera-from-world (OpenCV) → world-from-camera (Y-up).
        wfc_ocv = _cam_from_world_to_world_from_cam(cfw_ocv)  # world-from-camera in OpenCV
        wfc_yup = _opencv_rdf_to_yup_poses(wfc_ocv)  # world-from-camera in Y-up

        # Ground plane: fit in Y-up frame.
        ground_plane = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        if self._estimate_ground:
            ground_plane = _estimate_ground_plane_yup(wp_yup)

        # Drivable mask: heuristic — low-height points in Y-up frame.
        drivable_mask = np.zeros(wp_yup.shape[1:3], dtype=bool)
        if self._estimate_drivable and wp_yup.size > 0:
            drivable_mask = _estimate_drivable_region_yup(wp_yup, ground_plane)

        # ---- Scale recovery: reverse VGGT's unit-average-distance normalization ----
        # VGGT training divides world_points by avg_scale (mean distance from first
        # camera).  We recover this scale from the ground plane (if camera height is
        # known) or from the camera baseline.  The recovered scale is applied to all
        # VGGT outputs so downstream modules work in real metres.
        from ..fusion.coord_utils import (
            recover_scale_from_ground,
            apply_scale_to_pointmap,
            apply_scale_to_poses,
        )

        scale_factor = 1.0
        scale_source = "none"

        if self._known_camera_height_m is not None and self._estimate_ground:
            scale_factor = recover_scale_from_ground(
                ground_plane, self._known_camera_height_m
            )
            scale_source = f"ground (h={self._known_camera_height_m}m)"
        # Fallback: try baseline method if we have ≥2 frames.
        elif len(images) >= 2:
            # Heuristic: assume ~1m baseline as a rough guess for hand-held capture.
            from ..fusion.coord_utils import recover_scale_from_baseline
            scale_factor = recover_scale_from_baseline(cfw_ocv, known_baseline_m=1.0)
            scale_source = "baseline (assumed 1m)"

        if scale_factor != 1.0 and scale_source != "none":
            wp_yup = apply_scale_to_pointmap(wp_yup, scale_factor)
            wfc_yup = apply_scale_to_poses(wfc_yup, scale_factor)
            wfc_ocv = apply_scale_to_poses(wfc_ocv, scale_factor)
            cfw_ocv = apply_scale_to_poses(cfw_ocv, scale_factor)
            logger.info(
                "VGGT scale recovered (%s): factor=%.3f → pointmap in real metres.",
                scale_source, scale_factor,
            )
        else:
            logger.info(
                "VGGT scale NOT recovered — output is in unit-scale (avg dist ≈ 1). "
                "Pass known_camera_height_m= to enable scale recovery."
            )

        n = len(images)
        logger.info(
            "VGGT inference: %d frames, pointmap %s (Y-up), ground=%s.",
            n, wp_yup.shape, tuple(ground_plane.round(3)),
        )
        # Record processed image dimensions for downstream pose consumers.
        h_proc, w_proc = processed[0].shape[1], processed[0].shape[2]
        return {
            "pointmap": wp_yup,                     # Y-up world points
            "camera_poses": wfc_yup,                # world-from-camera in Y-up
            "camera_poses_wfc_ocv": wfc_ocv,        # world-from-camera in OpenCV (for MVSplat)
            "camera_poses_cfw": cfw_ocv,            # camera-from-world in OpenCV (raw VGGT)
            "ground_plane": ground_plane,            # Y-up
            "drivable_mask": drivable_mask,          # Y-up
            "image_size_hw": (h_proc, w_proc),
            "scale_factor": scale_factor,
            "scale_source": scale_source,
        }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _resize_image(image: np.ndarray, max_res: int, patch_size: int = 14) -> np.ndarray:
    """Resize image so max(H, W) ≤ max_res, then round to patch_size multiples.

    VGGT uses a ViT backbone with patch_size=14.  Input dimensions must be
    divisible by 14 or the patch embedder asserts.

    Raises:
        ImportError: If cv2 (opencv-python) is not installed and resizing is needed.
    """
    h, w = image.shape[:2]
    scale = min(max_res / max(h, w), 1.0)
    if scale < 1.0:
        new_h, new_w = int(h * scale), int(w * scale)
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "opencv-python (cv2) is required for VGGT image preprocessing. "
                "Install with: pip install opencv-python"
            )
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = new_h, new_w
    # Round down to nearest patch_size multiple.
    h = (h // patch_size) * patch_size
    w = (w // patch_size) * patch_size
    if h != image.shape[0] or w != image.shape[1]:
        # cv2 already imported above if scale < 1.0; import here for the
        # case where only patch-alignment resize is needed.
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "opencv-python (cv2) is required for VGGT image preprocessing. "
                "Install with: pip install opencv-python"
            )
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    return image.transpose(2, 0, 1)  # HWC → CHW


def _pose_enc_to_matrices(pose_enc: np.ndarray) -> np.ndarray:
    """Convert VGGT pose encoding (S, 9) → (S, 4, 4) camera-from-world matrices.

    VGGT pose_enc format (absT_quaR_FoV, OpenCV convention):
      Indices 0:3 = translation T (3D)
      Indices 3:7 = rotation quaternion (xyzw, scalar-last)
      Indices 7:9 = field of view (h, w) — discarded

    The resulting 4×4 matrix maps world coords (OpenCV RDF) to camera coords (OpenCV RDF):
      X_cam = R @ X_world + T

    For world-from-camera (used by downstream), invert the result.
    """
    s = pose_enc.shape[0]
    T = pose_enc[:, :3]  # (S, 3)
    quat_xyzw = pose_enc[:, 3:7]  # (S, 4) — xyzw, scalar-last

    # Convert quaternion (xyzw, scalar-last) → rotation matrix (S, 3, 3).
    R = _quat_xyzw_to_mat(quat_xyzw)

    # Build camera-from-world 4×4: [R | T; 0 0 0 1].
    matrices = np.tile(np.eye(4, dtype=np.float32), (s, 1, 1))
    matrices[:, :3, :3] = R
    matrices[:, :3, 3] = T
    return matrices


def _quat_xyzw_to_mat(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert quaternions (xyzw, scalar-last) to rotation matrices.

    Args:
        quat_xyzw: (..., 4) quaternions with scalar (w) last.

    Returns:
        (..., 3, 3) rotation matrices.
    """
    i, j, k, r = np.split(quat_xyzw, 4, axis=-1)
    i, j, k, r = i[..., 0], j[..., 0], k[..., 0], r[..., 0]

    two_s = 2.0 / (i * i + j * j + k * k + r * r)

    o00 = 1 - two_s * (j * j + k * k)
    o01 = two_s * (i * j - k * r)
    o02 = two_s * (i * k + j * r)
    o10 = two_s * (i * j + k * r)
    o11 = 1 - two_s * (i * i + k * k)
    o12 = two_s * (j * k - i * r)
    o20 = two_s * (i * k - j * r)
    o21 = two_s * (j * k + i * r)
    o22 = 1 - two_s * (i * i + j * j)

    shape = quat_xyzw.shape[:-1] + (3, 3)
    R = np.stack([o00, o01, o02, o10, o11, o12, o20, o21, o22], axis=-1).reshape(shape)
    return R


def _cam_from_world_to_world_from_cam(cfw: np.ndarray) -> np.ndarray:
    """Invert camera-from-world [R|T] matrices to world-from-camera [R^T | -R^T@T].

    Args:
        cfw: (..., 4, 4) camera-from-world matrices.

    Returns:
        (..., 4, 4) world-from-camera matrices.
    """
    R = cfw[..., :3, :3]  # (..., 3, 3)
    T = cfw[..., :3, 3]  # (..., 3)
    Rt = np.swapaxes(R, -1, -2)  # transpose: R^T
    wfc = np.zeros_like(cfw)
    wfc[..., :3, :3] = Rt
    wfc[..., :3, 3] = -np.sum(Rt * T[..., None, :], axis=-1)  # -R^T @ T
    wfc[..., 3, 3] = 1.0
    return wfc


# --------------------------------------------------------------------------
# Coordinate conversion: OpenCV RDF → project Y-up
# --------------------------------------------------------------------------
#
# VGGT natively outputs world_points and poses in OpenCV convention:
#   +X = right, +Y = down, +Z = forward (RDF, right-handed)
#   First camera at origin, looking along +Z.
#
# The project uses a Y-up world frame (OpenGL-like):
#   +X = right, +Y = up, +Z = backward (right-handed)
#
# Conversion: R = diag(1, -1, -1) — 180° rotation around X axis.
#   X_yup = X_ocv
#   Y_yup = -Y_ocv  (up = opposite of down)
#   Z_yup = -Z_ocv  (backward = opposite of forward)
# --------------------------------------------------------------------------

_R_OCV_TO_YUP = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)


def _opencv_rdf_to_yup_points(points_ocv: np.ndarray) -> np.ndarray:
    """Convert points from OpenCV RDF to project Y-up frame.

    Args:
        points_ocv: (..., 3) points in OpenCV RDF (x-right, y-down, z-forward).

    Returns:
        (..., 3) points in Y-up (x-right, y-up, z-backward).
    """
    points = np.asarray(points_ocv, dtype=np.float32)
    out = np.zeros_like(points)
    out[..., 0] = points[..., 0]  # X stays
    out[..., 1] = -points[..., 1]  # -Y_ocv (down) → Y_yup (up)
    out[..., 2] = -points[..., 2]  # -Z_ocv (forward) → Z_yup (backward)
    return out


def _opencv_rdf_to_yup_poses(wfc_ocv: np.ndarray) -> np.ndarray:
    """Convert world-from-camera poses from OpenCV RDF world to Y-up world.

    Given E_wfc_ocv that maps OpenCV camera coords → OpenCV world coords,
    produce E_wfc_yup that maps the SAME camera coords → Y-up world coords.

    p_world_yup = R @ p_world_ocv = R @ (E_wfc_ocv @ p_cam) = (R @ E_wfc_ocv) @ p_cam
    So E_wfc_yup = R @ E_wfc_ocv.

    Args:
        wfc_ocv: (..., 4, 4) world-from-camera in OpenCV world frame.

    Returns:
        (..., 4, 4) world-from-camera in Y-up world frame.
    """
    R = _R_OCV_TO_YUP
    wfc = wfc_ocv.copy()
    wfc[..., :3, :3] = R @ wfc_ocv[..., :3, :3]
    wfc[..., :3, 3] = (R @ wfc_ocv[..., :3, 3, None])[..., 0]
    return wfc


# ------------------------------------------------------------------
# Ground estimation helpers (operate in Y-up frame)
# ------------------------------------------------------------------


def _estimate_ground_plane_yup(world_points: np.ndarray) -> np.ndarray:
    """Fit ground plane (a,b,c,d) to central-bottom region in Y-up frame.

    In Y-up: the ground is roughly aligned with the XZ plane (normal ≈ +Y).
    Samples the central-bottom region of the first-frame pointmap and fits
    a plane via SVD.
    """
    pm = world_points[0]  # (H, W, 3) — first frame, Y-up
    h, w = pm.shape[:2]
    # Central 60% width, bottom 40% height (near-ground region in image).
    cx_start, cx_end = int(w * 0.2), int(w * 0.8)
    cy_start, cy_end = int(h * 0.6), h
    region = pm[cy_start:cy_end, cx_start:cx_end].reshape(-1, 3)
    # Remove extreme outliers along Y axis (height in Y-up).
    height_vals = region[:, 1]
    q05, q95 = np.percentile(height_vals, [5, 95])
    inliers = region[(height_vals >= q05) & (height_vals <= q95)]
    if len(inliers) < 10:
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    # SVD fit: centroid + normal.
    centroid = inliers.mean(axis=0)
    _, _, vh = np.linalg.svd(inliers - centroid, full_matrices=False)
    normal = vh[2]  # smallest singular vector
    normal = normal / (np.linalg.norm(normal) + 1e-8)
    # Ensure normal points upward (Y+ in Y-up).
    if normal[1] < 0:
        normal = -normal
    d = -np.dot(normal, centroid)
    return np.array([*normal, d], dtype=np.float32)


def _estimate_drivable_region_yup(
    wp: np.ndarray, ground_plane: np.ndarray, height_thresh: float = 0.3
) -> np.ndarray:
    """Heuristic drivable mask: points within height_thresh of the ground plane.

    Operates in Y-up frame.
    """
    normal = ground_plane[:3]
    d = ground_plane[3]
    pm = wp[0]  # First frame pointmap in Y-up.
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

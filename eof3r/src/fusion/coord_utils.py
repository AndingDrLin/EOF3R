"""Coordinate-system conversion utilities.

3D World/Occupancy: Right-handed, Y-up (OpenGL convention)
BEV/Robot:          X-forward, Y-left, Z-up (ROS convention)

Y-up → Z-up conversion:
  yup: (x_right, y_up, z_backward)
  zup: (x_forward, y_left, z_up)

  x_zup = x_yup
  y_zup = -z_yup
  z_zup = y_yup
"""

from __future__ import annotations

import numpy as np


def yup_to_zup(points_yup: np.ndarray) -> np.ndarray:
    """Convert 3D points from Y-up (OpenGL) to Z-up (ROS/BEV) convention.

    Args:
        points_yup: (..., 3) array in Y-up coords (x-right, y-up, z-backward).

    Returns:
        (..., 3) array in Z-up coords (x-forward, y-left, z-up).
    """
    points = np.asarray(points_yup, dtype=np.float32)
    out = np.zeros_like(points)
    out[..., 0] = points[..., 0]  # x → x
    out[..., 1] = -points[..., 2]  # -z → y
    out[..., 2] = points[..., 1]  # y → z
    return out


def zup_to_yup(points_zup: np.ndarray) -> np.ndarray:
    """Convert 3D points from Z-up (ROS/BEV) to Y-up (OpenGL) convention.

    Args:
        points_zup: (..., 3) array in Z-up coords (x-forward, y-left, z-up).

    Returns:
        (..., 3) array in Y-up coords (x-right, y-up, z-backward).
    """
    points = np.asarray(points_zup, dtype=np.float32)
    out = np.zeros_like(points)
    out[..., 0] = points[..., 0]  # x → x
    out[..., 1] = points[..., 2]  # z → y
    out[..., 2] = -points[..., 1]  # -y → z
    return out


def convert_covariance_yup_to_zup(cov_yup: np.ndarray) -> np.ndarray:
    """Convert 3x3 covariance matrices from Y-up to Z-up frame.

    Uses rotation R: R @ p_yup = p_zup, so C_zup = R @ C_yup @ R^T.
    R = [[1,0,0], [0,0,-1], [0,1,0]] (maps Y-up to Z-up).

    Args:
        cov_yup: (..., 3, 3) covariance in Y-up frame.

    Returns:
        (..., 3, 3) covariance in Z-up frame.
    """
    cov = np.asarray(cov_yup, dtype=np.float32)
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    return R @ cov @ R.T


# ---------------------------------------------------------------------------
# Scale recovery — reverse VGGT's unit-average-distance normalization
# ---------------------------------------------------------------------------
# VGGT training normalizes world_points so avg ||p|| ≈ 1 from the first
# camera (training/train_utils/normalization.py:100-107).  At inference the
# model outputs points in this unit-scale frame.  We recover the real scale
# via one of several anchors and multiply the VGGT outputs to restore metres.
# ---------------------------------------------------------------------------


def recover_scale_from_ground(
    ground_plane_yup: np.ndarray,
    known_camera_height_m: float = 1.5,
) -> float:
    """Recover scale factor from VGGT-estimated ground plane.

    In Y-up frame, the ground plane is (a,b,c,d) with ax+by+cz+d=0.
    The camera is at origin.  The perpendicular distance from origin to the
    plane is |d| / ||(a,b,c)||  (since origin satisfies ax+by+cz+d = d).

    VGGT outputs this distance in unit-scale.  If the real camera is mounted
    at a known height above the ground, the scale factor is:
        scale = known_height / vggt_height

    Args:
        ground_plane_yup: (4,) float32 (a, b, c, d) in Y-up.
        known_camera_height_m: Real camera height above ground (metres).

    Returns:
        Scale factor (multiply VGGT unit-scale coords to get metres).
    """
    normal = ground_plane_yup[:3]
    d = ground_plane_yup[3]
    norm = float(np.linalg.norm(normal))
    if norm < 1e-8:
        return 1.0
    vggt_height = abs(d) / norm
    if vggt_height < 1e-6:
        return 1.0
    return known_camera_height_m / vggt_height


def recover_scale_from_baseline(
    poses_cfw_ocv: np.ndarray,
    known_baseline_m: float = 1.0,
) -> float:
    """Recover scale from known camera baseline.

    VGGT camera poses (camera-from-world) contain the camera positions in
    unit-scale.  The baseline between the first two cameras:
        b_vggt = ||T_1 - T_0||
    If we know the real baseline (e.g. from wheel odometry or stereo rig):
        scale = known_baseline / b_vggt

    Args:
        poses_cfw_ocv: (N, 4, 4) camera-from-world in OpenCV, unit-scale.
        known_baseline_m: Real distance between first two cameras (metres).

    Returns:
        Scale factor.
    """
    if len(poses_cfw_ocv) < 2:
        return 1.0
    # Camera center in world coords: C = -R^T @ T for camera-from-world [R|T].
    R0 = poses_cfw_ocv[0, :3, :3]
    T0 = poses_cfw_ocv[0, :3, 3]
    C0 = -R0.T @ T0

    R1 = poses_cfw_ocv[1, :3, :3]
    T1 = poses_cfw_ocv[1, :3, 3]
    C1 = -R1.T @ T1

    vggt_baseline = float(np.linalg.norm(C1 - C0))
    if vggt_baseline < 1e-6:
        return 1.0
    return known_baseline_m / vggt_baseline


def recover_scale_from_depth(
    vggt_pointmap_yup: np.ndarray,
    depth_map_m: np.ndarray,
    K: np.ndarray,
    sample_ratio: float = 0.01,
) -> float:
    """Recover scale by comparing VGGT depths with measured (LiDAR/RGBD) depths.

    For a sample of pixels, compare the Z-coordinate of VGGT's 3D points
    (projected into camera space) with the measured depth at the same pixel.

    Args:
        vggt_pointmap_yup: (H, W, 3) first-frame VGGT pointmap in Y-up.
        depth_map_m: (H, W) measured depth in metres (LiDAR / RGBD).
        K: (3, 3) camera intrinsics.
        sample_ratio: Fraction of pixels to sample.

    Returns:
        Scale factor.
    """
    import numpy as np

    h, w = vggt_pointmap_yup.shape[:2]
    # Convert VGGT Y-up → OpenCV camera space for depth comparison.
    # VGGT pointmap is in OpenCV world frame. To get camera-space depth:
    # The first camera is at origin in VGGT's frame, looking +Z_ocv.
    # Depth = Z_ocv coordinate.
    # We have the pointmap in Y-up.  Y-up → OpenCV: R_yup_to_ocv = R_ocv_to_yup^T.
    _R_OCV_TO_YUP = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
    _R_YUP_TO_OCV = _R_OCV_TO_YUP.T  # Same matrix (symmetric).
    pts_ocv = (vggt_pointmap_yup.reshape(-1, 3) @ _R_YUP_TO_OCV.T).reshape(h, w, 3)
    vggt_depth = pts_ocv[..., 2]  # Z in OpenCV = depth

    # Sample pixels where both have valid depth.
    valid = (depth_map_m > 0.1) & (depth_map_m < 100) & (vggt_depth > 1e-3)
    n_valid = int(valid.sum())
    if n_valid < 100:
        return 1.0

    n_sample = max(100, int(n_valid * sample_ratio))
    indices = np.flatnonzero(valid)
    sampled = np.random.RandomState(42).choice(indices, n_sample, replace=False)
    rows, cols = np.unravel_index(sampled, (h, w))

    real_depths = depth_map_m[rows, cols]
    vggt_depths = vggt_depth[rows, cols]

    # Robust scale: median ratio (avoids outlier corruption).
    ratios = real_depths / np.maximum(vggt_depths, 1e-6)
    scale = float(np.median(ratios))
    return max(scale, 0.01)


def apply_scale_to_pointmap(pointmap: np.ndarray, scale: float) -> np.ndarray:
    """Scale a VGGT pointmap by a factor (reverses unit normalization)."""
    return pointmap.astype(np.float32) * scale


def apply_scale_to_poses(poses_cfw: np.ndarray, scale: float) -> np.ndarray:
    """Scale camera-from-world poses (translation part only)."""
    out = poses_cfw.copy()
    out[..., :3, 3] *= scale
    return out


# ---------------------------------------------------------------------------
# Dynamic BEV grid bounds
# ---------------------------------------------------------------------------


def compute_bev_bounds_from_data(
    points_yup: np.ndarray,
    height_range_yup: tuple[float, float] = (-0.5, 2.0),
    margin_ratio: float = 0.15,
    target_cells: int = 400,
) -> tuple[list[float], float]:
    """Compute BEV grid bounds and resolution from actual data extent.

    After Y-up → Z-up conversion, the BEV grid uses X (forward) and Y (left).
    This function filters points by height (Y in Y-up), computes the XZ extent,
    and derives bounds + resolution to achieve the target cell count.

    Args:
        points_yup: (N, 3) points in Y-up (x=right, y=up, z=backward).
        height_range_yup: (y_min, y_max) in Y-up for height filtering.
        margin_ratio: Fraction of range to add as margin on each side.
        target_cells: Desired grid cells per dimension.

    Returns:
        (bev_range [x_min, y_min, x_max, y_max], resolution_m_per_cell).
    """
    y = points_yup[:, 1]
    mask = (y >= height_range_yup[0]) & (y <= height_range_yup[1])
    if mask.sum() < 10:
        # Fallback: use all points.
        mask = np.ones(len(points_yup), dtype=bool)

    pts = points_yup[mask]
    # In Y-up: x=right, z=backward → BEV X=right, BEV Y=backward.
    # After Y-up→Z-up: x_zup = x_yup, y_zup = -z_yup.
    x_zup = pts[:, 0]
    y_zup = -pts[:, 2]

    x_min, x_max = float(x_zup.min()), float(x_zup.max())
    y_min, y_max = float(y_zup.min()), float(y_zup.max())

    x_range = x_max - x_min
    y_range = y_max - y_min
    if x_range < 0.1:
        x_range = 10.0
        x_min, x_max = -5.0, 5.0
    if y_range < 0.1:
        y_range = 10.0
        y_min, y_max = -5.0, 5.0

    x_margin = x_range * margin_ratio
    y_margin = y_range * margin_ratio

    resolution = max(x_range + 2 * x_margin, y_range + 2 * y_margin) / target_cells
    resolution = max(resolution, 0.01)  # floor at 1cm

    return (
        [x_min - x_margin, y_min - y_margin, x_max + x_margin, y_max + y_margin],
        resolution,
    )

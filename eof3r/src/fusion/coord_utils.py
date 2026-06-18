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

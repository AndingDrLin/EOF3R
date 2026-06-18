"""BEV projector: converts 3D Gaussians + background pointmap to BEV occupancy grid.

Coordinate flow:
  - Foreground Gaussians arrive in Y-up (OpenGL).
  - Background pointmap arrives in Y-up (OpenGL).
  - Conversion to Z-up (ROS/BEV) happens at this boundary.
  - Output BEV grid is in Z-up, X-forward, Y-left.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .coord_utils import yup_to_zup


@dataclass
class BEVGrid:
    """Normalised BEV occupancy grid."""

    data: np.ndarray  # (H, W) float32 occupancy values
    resolution: float  # meters/cell
    x_range: tuple[float, float]  # (x_min, x_max) in meters
    y_range: tuple[float, float]  # (y_min, y_max) in meters


class BEVProjector:
    """Convert 3D representations to BEV occupancy grids.

    All methods that accept `config` use keys from config.fusion:
      - bev_resolution (float, m/cell)
      - bev_range (list[float], [x_min, y_min, x_max, y_max])
      - height_filter (list[float], [z_min, z_max] in Z-up)
      - gaussian_to_occupancy (str, "max" | "sum" | "threshold")
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        self._resolution = cfg.get("bev_resolution", 0.05)
        rng = cfg.get("bev_range", [-10, -10, 10, 10])
        self._x_range = (rng[0], rng[2])
        self._y_range = (rng[1], rng[3])
        self._height_range = cfg.get("height_filter", (-0.5, 2.0))
        self._agg_mode = cfg.get("gaussian_to_occupancy", "max")
        self._alpha_threshold = cfg.get("alpha_threshold", 0.3)

    @property
    def grid_shape(self) -> tuple[int, int]:
        """(height, width) in cells."""
        w = int((self._x_range[1] - self._x_range[0]) / self._resolution)
        h = int((self._y_range[1] - self._y_range[0]) / self._resolution)
        return (h, w)

    # ------------------------------------------------------------------
    # Main projection
    # ------------------------------------------------------------------

    def project_gaussians_to_bev(
        self,
        means: np.ndarray,
        opacities: np.ndarray,
        scales: np.ndarray | None = None,
        covariances: np.ndarray | None = None,
        config: dict | None = None,
    ) -> np.ndarray:
        """Project foreground Gaussians to BEV occupancy grid.

        Args:
            means: (N, 3) Gaussian centers in Y-up (x-right, y-up, z-backward).
            opacities: (N,) opacity values [0, 1].
            scales: (N, 3) scale per axis (Y-up). If None, uses fixed 0.05.
            covariances: (N, 3, 3) covariances (Y-up). If None, derived from scales.
            config: Optional overrides (same keys as __init__ config).

        Returns:
            (H, W) float32 BEV occupancy grid in Z-up (X-forward, Y-left).
        """
        self._apply_config_overrides(config)

        # Convert to Z-up.
        means_zup = yup_to_zup(means)

        if scales is None:
            scales_zup = np.full((means.shape[0], 3), 0.05, dtype=np.float32)
        else:
            scales_zup = yup_to_zup(scales)
            # scales are positive radii, not signed vectors, so take absolute.
            scales_zup = np.abs(scales_zup)

        # Height filter in Z-up (z is height).
        z = means_zup[:, 2]
        mask = (z >= self._height_range[0]) & (z <= self._height_range[1])
        m = means_zup[mask]
        o = opacities[mask]
        s = scales_zup[mask]

        h, w = self.grid_shape
        bev = np.zeros((h, w), dtype=np.float32)

        x_min, x_max = self._x_range
        y_min, y_max = self._y_range

        for i in range(len(m)):
            mx, my, mz = m[i]
            sx, sy, sz = s[i]
            col = int((mx - x_min) / self._resolution)
            row = int((my - y_min) / self._resolution)
            if 0 <= row < h and 0 <= col < w:
                r = max(sx, sy) * 3.0
                rc = max(1, int(r / self._resolution))
                for dr in range(-rc, rc + 1):
                    for dc in range(-rc, rc + 1):
                        nr, nc = row + dr, col + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            dist = np.sqrt(dr**2 + dc**2) * self._resolution
                            if dist <= r:
                                weight = o[i] * np.exp(-0.5 * (dist / max(r, 1e-6)) ** 2)
                                if self._agg_mode == "max":
                                    bev[nr, nc] = max(bev[nr, nc], weight)
                                elif self._agg_mode == "sum":
                                    bev[nr, nc] += weight
                                else:  # threshold
                                    bev[nr, nc] = max(bev[nr, nc], float(weight > self._alpha_threshold))

        print(
            f"[BEVProjector] Projected {len(m)} Gaussians → BEV {bev.shape}, "
            f"occupied (>{self._alpha_threshold}): {(bev > self._alpha_threshold).sum()}"
        )
        return bev

    # ------------------------------------------------------------------
    # Semantic projection
    # ------------------------------------------------------------------

    def project_semantic_to_bev(
        self,
        means: np.ndarray,
        semantic_labels: np.ndarray,
        opacities: np.ndarray | None = None,
        config: dict | None = None,
    ) -> np.ndarray:
        """Project per-Gaussian semantic labels to BEV (argmax per cell).

        Args:
            means: (N, 3) Gaussian centers in Y-up.
            semantic_labels: (N,) integer class indices.
            opacities: (N,) optional opacity for weighting. If None, equal weight.
            config: Optional overrides.

        Returns:
            (H, W) int32 semantic BEV grid (most frequent class per cell).
        """
        self._apply_config_overrides(config)

        means_zup = yup_to_zup(means)
        z = means_zup[:, 2]
        mask = (z >= self._height_range[0]) & (z <= self._height_range[1])
        m = means_zup[mask]
        labs = semantic_labels[mask]
        wgt = opacities[mask] if opacities is not None else np.ones(len(m), dtype=np.float32)

        h, w_cells = self.grid_shape
        x_min, x_max = self._x_range
        y_min, y_max = self._y_range

        # Accumulate weighted votes per cell.
        num_classes = int(labs.max()) + 1
        votes = np.zeros((h, w_cells, num_classes), dtype=np.float32)

        for i in range(len(m)):
            mx, my, mz = m[i]
            col = int((mx - x_min) / self._resolution)
            row = int((my - y_min) / self._resolution)
            if 0 <= row < h and 0 <= col < w_cells:
                votes[row, col, labs[i]] += wgt[i]

        semantic_bev = np.argmax(votes, axis=-1).astype(np.int32)
        return semantic_bev

    # ------------------------------------------------------------------
    # Foreground + Background fusion
    # ------------------------------------------------------------------

    def fuse_foreground_background(
        self,
        fg_bev: np.ndarray,
        bg_occupancy: np.ndarray | None = None,
        bg_drivable: np.ndarray | None = None,
        config: dict | None = None,
    ) -> np.ndarray:
        """Fuse foreground and background BEV occupancy.

        Strategy (configurable via gaussian_to_occupancy):
          - "max": take per-cell maximum (fg obstacles override bg drivable).
          - "sum": add contributions.
          - "threshold": binary OR after thresholding.

        Args:
            fg_bev: (H, W) foreground occupancy from project_gaussians_to_bev.
            bg_occupancy: (H, W) background occupancy (resized if needed).
            bg_drivable: (H, W) drivable mask (True = drivable, False = obstacle). Resized.
            config: Optional overrides.

        Returns:
            (H, W) fused BEV occupancy grid.
        """
        self._apply_config_overrides(config)

        h, w = self.grid_shape
        fused = fg_bev.copy()

        # Resize background inputs to match grid if needed.
        if bg_occupancy is not None:
            bg = self._resize_to_grid(bg_occupancy, h, w)
            if self._agg_mode == "max":
                fused = np.maximum(fused, bg)
            elif self._agg_mode == "sum":
                fused = fused + bg
            else:
                fused = np.maximum(fused, (bg > self._alpha_threshold).astype(np.float32))

        if bg_drivable is not None:
            dm = self._resize_to_grid(bg_drivable.astype(np.float32), h, w)
            # Non-drivable regions = high occupancy.
            obstacle = (1.0 - dm) * 0.7
            if self._agg_mode == "max":
                fused = np.maximum(fused, obstacle)
            elif self._agg_mode == "sum":
                fused = fused + obstacle
            else:
                fused = np.maximum(fused, (obstacle > self._alpha_threshold).astype(np.float32))

        fused = np.clip(fused, 0.0, 1.0)
        print(
            f"[BEVProjector] Fused BEV: shape={fused.shape}, "
            f"max={fused.max():.3f}, occupied={ (fused > self._alpha_threshold).sum()}"
        )
        return fused

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_config_overrides(self, config: dict | None) -> None:
        if config is None:
            return
        if "bev_resolution" in config:
            self._resolution = config["bev_resolution"]
        if "bev_range" in config:
            rng = config["bev_range"]
            self._x_range = (rng[0], rng[2])
            self._y_range = (rng[1], rng[3])
        if "height_filter" in config:
            self._height_range = tuple(config["height_filter"])
        if "gaussian_to_occupancy" in config:
            self._agg_mode = config["gaussian_to_occupancy"]
        if "alpha_threshold" in config:
            self._alpha_threshold = config["alpha_threshold"]

    @staticmethod
    def _resize_to_grid(arr: np.ndarray, h: int, w: int) -> np.ndarray:
        """Simple resize via numpy slicing/cropping/padding — no OpenCV dependency."""
        ah, aw = arr.shape
        if ah == h and aw == w:
            return arr.astype(np.float32)
        result = np.zeros((h, w), dtype=np.float32)
        copy_h = min(h, ah)
        copy_w = min(w, aw)
        result[:copy_h, :copy_w] = arr[:copy_h, :copy_w]
        return result

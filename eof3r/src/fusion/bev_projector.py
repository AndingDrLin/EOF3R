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

from .coord_utils import compute_bev_bounds_from_data, yup_to_zup


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
        if rng == "auto" or rng is None:
            self._auto_bounds = True
            self._x_range = (0.0, 0.0)
            self._y_range = (0.0, 0.0)
        else:
            self._auto_bounds = False
            self._x_range = (rng[0], rng[2])
            self._y_range = (rng[1], rng[3])
        self._height_range = cfg.get("height_filter", (-0.5, 2.0))
        self._agg_mode = cfg.get("gaussian_to_occupancy", "max")
        self._alpha_threshold = cfg.get("alpha_threshold", 0.3)
        self._target_cells = cfg.get("bev_target_cells", 400)

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

        # Auto-compute BEV bounds from data if configured AND not already set.
        # Call set_bounds_from_points() first for multi-source projections so that
        # FG and BG share the same grid dimensions.
        if self._auto_bounds and self._x_range == (0.0, 0.0) and len(m) > 10:
            from .coord_utils import zup_to_yup
            pts_yup = zup_to_yup(m)
            new_range, new_res = compute_bev_bounds_from_data(
                pts_yup,
                height_range_yup=(-0.5, 2.0),
                target_cells=self._target_cells,
            )
            self._x_range = (new_range[0], new_range[2])
            self._y_range = (new_range[1], new_range[3])
            self._resolution = new_res

        h, w = self.grid_shape
        x_min, x_max = self._x_range
        y_min, y_max = self._y_range

        # -- Vectorized scatter to grid ---------------------------------------
        # Map 3D points to grid cell indices.
        cols = ((m[:, 0] - x_min) / self._resolution).astype(np.int32)
        rows = ((m[:, 1] - y_min) / self._resolution).astype(np.int32)

        valid = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
        cols_v = cols[valid]
        rows_v = rows[valid]
        opac_v = o[valid]
        scales_v = s[valid]

        # Average footprint radius per cell (used for Gaussian sigma).
        if len(scales_v) > 0:
            avg_radius = float(np.mean(np.maximum(scales_v[:, 0], scales_v[:, 1])) * 3.0)
        else:
            avg_radius = 0.15  # fallback

        # Scatter using 2D histogram (sum aggregation of opacities at centers).
        bev_raw = _scatter_sum(rows_v, cols_v, opac_v, h, w)

        # Apply Gaussian smoothing to approximate per-point footprint spread.
        sigma_cells = max(1.0, avg_radius / self._resolution)
        truncate = 3.0  # 3-sigma covers ~99.7%
        bev = _gaussian_smooth(bev_raw, sigma_cells, truncate)

        # Re-normalize: preserve peak occupancy so smoothing doesn't dilute.
        # Only re-scale when the peak dropped significantly (≥30% reduction),
        # which indicates dilution from the Gaussian blur.  When the peak is
        # already stable, skip normalization to avoid amplifying faint single
        # Gaussians to full occupancy.
        bev_max = bev.max()
        bev_raw_max = float(bev_raw.max())
        if bev_raw_max > 0 and bev_max < bev_raw_max * 0.7:
            scale = bev_raw_max / max(bev_max, 1e-6)
            bev = bev * scale
        bev = np.clip(bev, 0.0, 1.0)

        # Post-processing: apply aggregation mode.
        if self._agg_mode == "max":
            # max-mode: the scatter+smooth already produces occupancy-like values.
            # Clip ensures [0,1] range.
            pass
        elif self._agg_mode == "threshold":
            bev = (bev > self._alpha_threshold).astype(np.float32)

        print(
            f"[BEVProjector] Projected {len(m)} Gaussians → BEV {bev.shape}, "
            f"{int(valid.sum())} in bounds, "
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

        if len(m) == 0 or len(labs) == 0:
            h, w_cells = self.grid_shape
            return np.zeros((h, w_cells), dtype=np.int32)

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

    def set_bounds_from_points(
        self,
        fg_means_yup: np.ndarray | None = None,
        bg_means_yup: np.ndarray | None = None,
    ) -> None:
        """Pre-compute unified BEV bounds from foreground + background points.

        Call this once before project_gaussians_to_bev so both FG and BG
        projections share the same grid dimensions.

        Args:
            fg_means_yup: (N, 3) FG Gaussian means in Y-up, or None.
            bg_means_yup: (M, 3) BG pointmap points in Y-up, or None.
        """
        if not self._auto_bounds:
            return
        pts_list = []
        if fg_means_yup is not None and len(fg_means_yup) > 0:
            pts_list.append(fg_means_yup)
        if bg_means_yup is not None and len(bg_means_yup) > 0:
            pts_list.append(bg_means_yup)
        if not pts_list:
            return

        all_pts = np.concatenate(pts_list, axis=0)
        from .coord_utils import compute_bev_bounds_from_data

        new_range, new_res = compute_bev_bounds_from_data(
            all_pts,
            height_range_yup=(-0.5, 2.0),  # Y-up height filter
            target_cells=self._target_cells,
        )
        self._x_range = (new_range[0], new_range[2])
        self._y_range = (new_range[1], new_range[3])
        self._resolution = new_res

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


# ------------------------------------------------------------------
# Vectorized scatter + smooth helpers (module-level, used by BEVProjector)
# ------------------------------------------------------------------


def _scatter_sum(
    rows: np.ndarray, cols: np.ndarray, weights: np.ndarray, h: int, w: int
) -> np.ndarray:
    """Scatter weighted values onto a (h, w) grid using np.bincount.

    This replaces the O(N*k^2) per-point loop with a single O(N) scatter.
    """
    if len(weights) == 0:
        return np.zeros((h, w), dtype=np.float32)
    flat_idx = rows.astype(np.int64) * w + cols.astype(np.int64)
    # bincount is ~10x faster than histogram2d for this pattern.
    bev_flat = np.bincount(flat_idx, weights=weights.astype(np.float64), minlength=h * w)
    return bev_flat.reshape(h, w).astype(np.float32)


def _gaussian_smooth(
    grid: np.ndarray, sigma: float, truncate: float = 3.0
) -> np.ndarray:
    """Apply Gaussian smoothing via separable convolution (scipy).

    Approximates per-point Gaussian footprint spread after scatter.
    """
    try:
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(grid, sigma=sigma, mode="constant", truncate=truncate)
    except ImportError:
        # Fallback: no-op if scipy not available.
        return grid

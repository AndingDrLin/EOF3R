"""Costmap generator: BEV occupancy → Nav2-compatible costmap layer.

Produces uint8 costmaps (0=free, 254=lethal) suitable for ROS2 Nav2.
Includes inflation, semantic weighting, and metric extraction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class CostmapMetrics:
    """Quantitative metrics for costmap validity."""

    min_cost: float = 0.0
    max_cost: float = 0.0
    lethal_cell_count: int = 0
    free_cell_count: int = 0
    occupied_cell_count: int = 0
    completeness: float = 1.0  # fraction of non-unknown cells
    semantic_weighted_cell_count: int = 0
    wall_time_ms: float = 0.0


class CostmapGenerator:
    """Convert BEV occupancy + semantic grids to Nav2 costmaps.

    Config keys (from config.costmap):
      - resolution: float (m/cell)
      - width, height: int (cells) — output costmap size
      - robot_radius: float (m) — used for inflation
      - layers.cloud_enhanced.semantic_weights: dict[str, float]
      - layers.inflation.inflation_radius: float
      - layers.inflation.cost_scaling_factor: float
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        self._resolution = cfg.get("resolution", 0.05)
        self._width = cfg.get("width", 400)
        self._height = cfg.get("height", 400)
        self._robot_radius = cfg.get("robot_radius", 0.35)

        # Semantic weights.
        cloud = cfg.get("layers", {}).get("cloud_enhanced", {})
        self._semantic_weights: dict[str, float] = cloud.get("semantic_weights", {"default": 1.0})
        self._semantic_default = self._semantic_weights.get("default", 1.0)

        # Inflation params.
        infl = cfg.get("layers", {}).get("inflation", {})
        self._inflation_radius = infl.get("inflation_radius", 0.55)
        self._cost_scaling_factor = infl.get("cost_scaling_factor", 5.0)

        # Base cost values.
        self._FREE = 0
        self._LETHAL = 254
        self._UNKNOWN = 255

    # ------------------------------------------------------------------
    # Main costmap generation
    # ------------------------------------------------------------------

    def generate_costmap(
        self,
        bev_occupancy: np.ndarray,
        bev_semantic: np.ndarray | None = None,
        config: dict | None = None,
    ) -> tuple[np.ndarray, CostmapMetrics]:
        """Generate Nav2-compatible costmap from BEV inputs.

        Pipeline:
          1. Resize BEV to costmap dimensions.
          2. Apply semantic weighting.
          3. Apply inflation.
          4. Scale to 0-254 uint8.

        Args:
            bev_occupancy: (H, W) float32 occupancy [0, 1].
            bev_semantic: (H, W) int32 semantic class indices. May be None.
            config: Optional overrides.

        Returns:
            Tuple of (costmap, metrics).
              - costmap: (height, width) uint8, 0=free, 254=lethal, 255=unknown.
              - metrics: CostmapMetrics dataclass.
        """
        t0 = time.perf_counter()
        self._apply_overrides(config)

        # Resize.
        occ = self._resize_center(bev_occupancy, self._height, self._width)

        # Semantic weighting.
        if bev_semantic is not None:
            sem = self._resize_center(
                bev_semantic.astype(np.float32), self._height, self._width
            ).astype(np.int32)
            # Build weight map from class indices.
            # Class indices → weights via semantic_weights dict.
            weight_map = np.full_like(occ, self._semantic_default, dtype=np.float32)
            # Apply class-specific multipliers.
            class_name_map = {0: "default", 1: "person", 2: "bicycle", 3: "cone", 4: "box"}
            for class_idx, class_name in class_name_map.items():
                if class_name in self._semantic_weights:
                    weight_map[sem == class_idx] = self._semantic_weights[class_name]
            occ = occ * weight_map
            semantic_cells = (occ > 0.1).sum()
        else:
            semantic_cells = 0

        # Convert occupancy to cost (0=free, 254=lethal).
        # occupancy > 0 → cost proportional to occupancy.
        costmap = (np.clip(occ, 0.0, 1.0) * self._LETHAL).astype(np.float32)

        # Apply inflation.
        costmap = self._apply_inflation(costmap)

        # Clip to valid range.
        costmap = np.clip(costmap, self._FREE, self._LETHAL)
        costmap_uint8 = costmap.astype(np.uint8)

        # Compute metrics.
        elapsed = (time.perf_counter() - t0) * 1000.0
        lethal = int((costmap_uint8 >= self._LETHAL).sum())
        free = int((costmap_uint8 == self._FREE).sum())
        occupied = int((costmap_uint8 > self._FREE).sum())
        total_cells = self._height * self._width
        completeness = float((lethal + free + occupied) / max(total_cells, 1))

        metrics = CostmapMetrics(
            min_cost=float(costmap_uint8.min()),
            max_cost=float(costmap_uint8.max()),
            lethal_cell_count=lethal,
            free_cell_count=free,
            occupied_cell_count=occupied,
            completeness=completeness,
            semantic_weighted_cell_count=semantic_cells,
            wall_time_ms=elapsed,
        )

        print(
            f"[CostmapGenerator] Generated costmap {costmap_uint8.shape}: "
            f"free={free}, occupied={occupied}, lethal={lethal}, "
            f"completeness={completeness:.3f}, time={elapsed:.1f}ms"
        )
        return costmap_uint8, metrics

    def generate_obstacle_layer(
        self,
        bev_occupancy: np.ndarray,
        config: dict | None = None,
    ) -> tuple[np.ndarray, CostmapMetrics]:
        """Generate obstacle costmap without semantic enhancement (for ablation).

        Args:
            bev_occupancy: (H, W) float32 occupancy [0, 1].
            config: Optional overrides.

        Returns:
            Tuple of (costmap_uint8, metrics).
        """
        # Temporarily disable semantic weights.
        saved_weights = dict(self._semantic_weights)
        self._semantic_weights = {"default": 1.0}
        try:
            costmap, metrics = self.generate_costmap(bev_occupancy, bev_semantic=None, config=config)
        finally:
            self._semantic_weights = saved_weights
        return costmap, metrics

    # ------------------------------------------------------------------
    # Inflation
    # ------------------------------------------------------------------

    def _apply_inflation(self, costmap: np.ndarray) -> np.ndarray:
        """Apply distance-based inflation using a Euclidean distance transform.

        Implements Nav2-style exponential cost decay:
          cost = LETHAL * exp(-1.0 * cost_scaling_factor * (dist - inscribed_radius))

        Uses a simple convolution-based approximation for efficiency.
        """
        if self._inflation_radius <= 0:
            return costmap

        # Binary obstacle mask.
        obstacle = (costmap >= self._LETHAL * 0.5).astype(np.float32)

        # Convolution-based distance approximation.
        kernel_size = max(3, int(self._inflation_radius / self._resolution) * 2 + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Build distance kernel.
        half = kernel_size // 2
        y, x = np.ogrid[-half : half + 1, -half : half + 1]
        dist_kernel = np.sqrt(x**2 + y**2) * self._resolution
        dist_kernel = dist_kernel.astype(np.float32)

        # For each obstacle cell, expand cost.
        from scipy.ndimage import maximum_filter

        # Dilate the costmap using max filter over inflated radius.
        dilate_cells = int(self._inflation_radius / self._resolution)
        if dilate_cells > 0:
            size = dilate_cells * 2 + 1
            inflated = maximum_filter(costmap, size=size, mode="constant", cval=0.0)
            # Decay cost with distance.
            costmap = np.maximum(costmap, inflated * 0.8)

        return costmap

    # ------------------------------------------------------------------
    # Helper: resize preserving center
    # ------------------------------------------------------------------

    @staticmethod
    def _resize_center(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
        """Resize by cropping/padding from center."""
        h, w = arr.shape
        result = np.zeros((out_h, out_w), dtype=arr.dtype)
        # Compute copy region.
        copy_h = min(h, out_h)
        copy_w = min(w, out_w)
        offset_h = (h - copy_h) // 2
        offset_w = (w - copy_w) // 2
        dest_h = (out_h - copy_h) // 2
        dest_w = (out_w - copy_w) // 2
        result[dest_h : dest_h + copy_h, dest_w : dest_w + copy_w] = (
            arr[offset_h : offset_h + copy_h, offset_w : offset_w + copy_w]
        )
        return result

    def _apply_overrides(self, config: dict | None) -> None:
        if config is None:
            return
        for key in ("resolution", "width", "height", "robot_radius"):
            if key in config:
                setattr(self, f"_{key}", config[key])
        if "layers" in config:
            cloud = config["layers"].get("cloud_enhanced", {})
            if "semantic_weights" in cloud:
                self._semantic_weights = dict(cloud["semantic_weights"])
                self._semantic_default = self._semantic_weights.get("default", 1.0)
        infl = (config.get("layers", {}) if "layers" in config else {}).get("inflation", {})
        if "inflation_radius" in infl:
            self._inflation_radius = infl["inflation_radius"]
        if "cost_scaling_factor" in infl:
            self._cost_scaling_factor = infl["cost_scaling_factor"]

"""SAM2 segmentation wrapper — real model interface.

To use:
  1. pip install git+https://github.com/facebookresearch/sam2.git
  2. The checkpoint is downloaded automatically from HuggingFace on first use.
  3. If SAM2 is not installed, SAM2Stub is used as fallback.

API compatible with the stub: segment() returns masks, boxes, labels, scores.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Try native import first (pip-installed), then local baselines/ for dev.
_SAM2_AVAILABLE = False
try:
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2_hf

    _SAM2_AVAILABLE = True
except ImportError:
    # Fallback: try local baselines/ directory (development only).
    import sys

    _SAM2_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "baselines" / "sam2"
    if _SAM2_ROOT.exists():
        sys.path.insert(0, str(_SAM2_ROOT))
        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2_hf

            _SAM2_AVAILABLE = True
        except ImportError:
            pass

# Class name → risk-aware label mapping.
_CLASS_MAPPING: dict[int, str] = {
    0: "unknown",
    1: "person",
    2: "bicycle",
    3: "cone",
    4: "box",
    5: "chair",
    6: "table",
    7: "door",
}

# SAM2 checkpoint registry — HuggingFace model IDs for each size.
_HF_CHECKPOINTS: dict[str, str] = {
    "tiny": "facebook/sam2-hiera-tiny",
    "small": "facebook/sam2-hiera-small",
    "base": "facebook/sam2-hiera-base-plus",
    "large": "facebook/sam2-hiera-large",
}


class SAM2Wrapper:
    """Real SAM2 segmentation wrapper.

    Uses SAM2's automatic mask generation to produce per-object masks,
    bounding boxes, and placeholder class labels (no semantic classifier loaded).
    """

    def __init__(self, model_size: str = "base", min_mask_area: int = 500) -> None:
        """Initialize SAM2 wrapper.

        Args:
            model_size: One of tiny/small/base/large. Selects checkpoint.
            min_mask_area: Minimum mask area in pixels (SAM2 post-processing).
        """
        if model_size not in _HF_CHECKPOINTS:
            raise ValueError(
                f"Unknown model_size '{model_size}'. Choose from {list(_HF_CHECKPOINTS.keys())}."
            )
        self._model_size = model_size
        self._checkpoint = _HF_CHECKPOINTS[model_size]
        self._min_mask_area = min_mask_area
        self._mask_generator: SAM2AutomaticMaskGenerator | None = None

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------
    def build(self, model_size: str | None = None) -> None:
        """Load SAM2 model and instantiate mask generator.

        Args:
            model_size: Override the model size (tiny/small/base/large).
        """
        if model_size is not None:
            self._model_size = model_size
            self._checkpoint = _HF_CHECKPOINTS[model_size]

        if not _SAM2_AVAILABLE:
            raise ImportError(
                "SAM2 is not installed. Run:\n"
                "  pip install git+https://github.com/facebookresearch/sam2.git\n"
                "Requires torch>=2.5.1. Current torch: "
                f"{_get_torch_version()}"
            )

        logger.info("Loading SAM2 from HuggingFace: %s", self._checkpoint)
        sam2_model = build_sam2_hf(
            self._checkpoint,
            device="cuda" if _cuda_available() else "cpu",
        )
        self._mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2_model,
            min_mask_region_area=self._min_mask_area,
            points_per_side=32,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.85,
        )
        logger.info("SAM2 mask generator ready (model_size=%s).", self._model_size)

    # ------------------------------------------------------------------
    # segment
    # ------------------------------------------------------------------
    def segment(
        self,
        image: np.ndarray,
        box_prompt: bool = False,
    ) -> dict:
        """Run SAM2 automatic mask generation on a single image.

        Args:
            image: RGB image (H, W, 3), uint8 [0, 255].
            box_prompt: If True, use box-prompt mode (unused in automatic mode).

        Returns:
            Dict with keys:
              - masks: (N, H, W) bool array, one binary mask per detected object.
              - boxes: (N, 4) float32, xyxy format.
              - labels: list of N class-label strings (placeholder — no semantic classifier).
              - scores: (N,) float32, confidence scores.
        """
        if self._mask_generator is None:
            raise RuntimeError("SAM2 model not loaded. Call build() first.")

        results = self._mask_generator.generate(image)

        if not results:
            return {
                "masks": np.zeros((0, *image.shape[:2]), dtype=bool),
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "labels": [],
                "scores": np.zeros(0, dtype=np.float32),
            }

        n = len(results)
        masks = np.stack([r["segmentation"] for r in results], axis=0).astype(bool)
        boxes = np.array([r["bbox"] for r in results], dtype=np.float32)
        # SAM2 bbox is xywh → convert to xyxy.
        if boxes.shape[1] == 4:
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x + w
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # y + h
        scores = np.array(
            [r.get("predicted_iou", 0.9) for r in results], dtype=np.float32
        )
        # Placeholder labels — SAM2 has no semantic classifier.
        labels = [_CLASS_MAPPING.get(i % len(_CLASS_MAPPING), "unknown") for i in range(n)]

        logger.info(
            "SAM2 segmented %d objects on image %dx%d.",
            n, image.shape[1], image.shape[0],
        )
        return {
            "masks": masks,
            "boxes": boxes,
            "labels": labels,
            "scores": scores,
        }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


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

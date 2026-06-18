"""SAM2 segmentation wrapper — real model interface.

Two modes:
  1. Automatic (default): SAM2AutomaticMaskGenerator — grid-based, no semantics.
  2. YOLO + box-prompt (recommended): YOLOv8-nano detects objects → SAM2 refines
     each bbox into a clean mask.  This gives semantic labels directly from
     YOLO's COCO classes and avoids the 65-fragment over-segmentation of
     automatic mode.

To use:
  pip install git+https://github.com/facebookresearch/sam2.git
  pip install ultralytics  # for YOLO mode

API: build() loads models; segment() runs segmentation; detect_and_segment()
     uses YOLO+SAM2 box-prompt for semantics.
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
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    _SAM2_AVAILABLE = True
except ImportError:
    import sys

    _SAM2_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "baselines" / "sam2"
    if _SAM2_ROOT.exists():
        sys.path.insert(0, str(_SAM2_ROOT))
        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2_hf
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            _SAM2_AVAILABLE = True
        except ImportError:
            pass

# ---- COCO class mapping (subset relevant to campus delivery) ----------------
# Full COCO 80-class mapping; we keep the ones most relevant for robot nav.
_COCO_CLASSES: dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    6: "train",
    7: "truck",
    9: "traffic_light",
    10: "fire_hydrant",
    11: "stop_sign",
    13: "parking_meter",
    14: "bench",
    15: "bird",
    16: "cat",
    17: "dog",
    18: "horse",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    31: "sports_ball",
    39: "bottle",
    41: "cup",
    43: "knife",
    44: "spoon",
    45: "bowl",
    47: "banana",
    48: "apple",
    49: "sandwich",
    50: "orange",
    51: "broccoli",
    52: "carrot",
    53: "hot_dog",
    54: "pizza",
    55: "donut",
    56: "chair",
    57: "couch",
    58: "potted_plant",
    59: "bed",
    60: "dining_table",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell_phone",
    72: "refrigerator",
    73: "book",
    75: "vase",
    76: "scissors",
    77: "teddy_bear",
    84: "book",
}

# Risk levels per class for costmap generation (0=free, 3=must_avoid).
_CLASS_RISK: dict[str, int] = {
    "person": 3,
    "bicycle": 2,
    "car": 3,
    "motorcycle": 3,
    "bus": 3,
    "truck": 3,
    "traffic_light": 2,
    "stop_sign": 2,
    "fire_hydrant": 2,
    "bench": 1,
    "backpack": 1,
    "suitcase": 1,
    "chair": 0,
    "couch": 0,
    "potted_plant": 0,
    "dining_table": 0,
    "tv": 0,
    "laptop": 0,
    "refrigerator": 0,
}

# SAM2 checkpoint registry — HuggingFace model IDs.
_HF_CHECKPOINTS: dict[str, str] = {
    "tiny": "facebook/sam2-hiera-tiny",
    "small": "facebook/sam2-hiera-small",
    "base": "facebook/sam2-hiera-base-plus",
    "large": "facebook/sam2-hiera-large",
}


class SAM2Wrapper:
    """SAM2 segmentation wrapper with optional YOLO frontend.

    Two operating modes:

    1. Automatic (segment): SAM2AutomaticMaskGenerator samples a 32×32 grid.
       Fast but produces many fragments, no semantic labels.
       Use for: quick testing, or when semantics come from elsewhere.

    2. YOLO + box-prompt (detect_and_segment): YOLOv8-nano detects objects
       → SAM2ImagePredictor refines each bbox into one clean mask.
       Gives: masks, boxes, COCO class labels, confidence scores, risk levels.
       Use for: production, navigation, costmap generation.
    """

    def __init__(
        self,
        model_size: str = "base",
        min_mask_area: int = 500,
        use_yolo: bool = True,
        yolo_model: str = "yolov8n.pt",  # nano: 6MB, small: "yolov8s.pt"
    ) -> None:
        """Initialize SAM2 wrapper.

        Args:
            model_size: One of tiny/small/base/large.
            min_mask_area: Minimum mask area in pixels (automatic mode only).
            use_yolo: If True, use YOLO detection as SAM2 frontend.
            yolo_model: YOLO model name or path. "yolov8n.pt" = nano (6MB).
        """
        if model_size not in _HF_CHECKPOINTS:
            raise ValueError(
                f"Unknown model_size '{model_size}'. Choose from {list(_HF_CHECKPOINTS.keys())}."
            )
        self._model_size = model_size
        self._checkpoint = _HF_CHECKPOINTS[model_size]
        self._min_mask_area = min_mask_area
        self._use_yolo = use_yolo
        self._yolo_model_name = yolo_model

        self._mask_generator: SAM2AutomaticMaskGenerator | None = None
        self._predictor: SAM2ImagePredictor | None = None
        self._yolo: object | None = None  # YOLO model

        # Cache the underlying SAM2 model so predictor and mask_gen can share it.
        self._sam2_model: object | None = None

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------
    def build(self, model_size: str | None = None) -> None:
        """Load SAM2 model + optional YOLO detector.

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

        # Load SAM2 model (shared between automatic and predictor).
        logger.info("Loading SAM2 from HuggingFace: %s", self._checkpoint)
        self._sam2_model = build_sam2_hf(
            self._checkpoint,
            device="cuda" if _cuda_available() else "cpu",
        )

        # Build automatic mask generator (fallback / backward compat).
        self._mask_generator = SAM2AutomaticMaskGenerator(
            model=self._sam2_model,
            min_mask_region_area=self._min_mask_area,
            points_per_side=32,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.85,
        )

        # Build predictor for box-prompt mode.
        self._predictor = SAM2ImagePredictor(self._sam2_model)

        # Load YOLO if requested.
        if self._use_yolo:
            self._init_yolo()

        logger.info(
            "SAM2 ready (model_size=%s, yolo=%s).",
            self._model_size,
            "enabled" if self._yolo is not None else "disabled",
        )

    # ------------------------------------------------------------------
    # segment (automatic mode — backward compatible)
    # ------------------------------------------------------------------
    def segment(
        self,
        image: np.ndarray,
        box_prompt: bool = False,
    ) -> dict:
        """Run SAM2 automatic mask generation on a single image.

        Args:
            image: RGB image (H, W, 3), uint8 [0, 255].
            box_prompt: Ignored in automatic mode (kept for API compat).

        Returns:
            Dict with keys:
              - masks: (N, H, W) bool array.
              - boxes: (N, 4) float32, xyxy format.
              - labels: list of N placeholder class-name strings.
              - scores: (N,) float32 confidence scores.
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
        labels = ["unknown"] * n

        logger.info(
            "SAM2 (automatic) segmented %d objects on %dx%d image.",
            n, image.shape[1], image.shape[0],
        )
        return {
            "masks": masks,
            "boxes": boxes,
            "labels": labels,
            "scores": scores,
        }

    # ------------------------------------------------------------------
    # detect_and_segment (YOLO + SAM2 box-prompt — recommended)
    # ------------------------------------------------------------------
    def detect_and_segment(
        self,
        image: np.ndarray,
        yolo_conf: float = 0.35,
        yolo_classes: list[int] | None = None,
    ) -> dict:
        """YOLO detection → SAM2 box-prompt refinement → masks + semantics.

        This is the recommended production path: YOLOv8-nano (6MB) detects
        objects with COCO class labels, then SAM2 refines each detection box
        into a clean binary mask.  Far fewer fragments than automatic mode
        (N objects vs 65+), and real semantic labels.

        Args:
            image: RGB image (H, W, 3), uint8 [0, 255].
            yolo_conf: YOLO confidence threshold (0–1).  Higher = fewer detections.
            yolo_classes: List of COCO class indices to detect, or None for all.

        Returns:
            Dict with keys:
              - masks: (N, H, W) bool — one clean mask per detected object.
              - boxes: (N, 4) float32 — xyxy boxes from YOLO.
              - labels: list of N COCO class-name strings ("person", "chair", …).
              - scores: (N,) float32 — combined confidence (YOLO × SAM2).
              - risk_levels: list of N int — risk level 0–3 for costmap.
              - coco_ids: list of N int — raw COCO class indices.
        """
        if self._predictor is None:
            raise RuntimeError("SAM2 model not loaded. Call build() first.")
        if self._yolo is None:
            raise RuntimeError(
                "YOLO not loaded. Set use_yolo=True in __init__ and call build()."
            )

        h, w = image.shape[:2]

        # ---- Step 1: YOLO detection ----
        yolo_results = self._yolo(image, conf=yolo_conf, classes=yolo_classes, verbose=False)
        yolo_boxes = yolo_results[0].boxes
        if yolo_boxes is None or len(yolo_boxes) == 0:
            logger.info("YOLO: no objects detected (conf=%.2f).", yolo_conf)
            return {
                "masks": np.zeros((0, h, w), dtype=bool),
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "labels": [],
                "scores": np.zeros(0, dtype=np.float32),
                "risk_levels": [],
                "coco_ids": [],
            }

        boxes_xyxy = yolo_boxes.xyxy.cpu().numpy().astype(np.float32)  # (N, 4)
        yolo_cls = yolo_boxes.cls.cpu().numpy().astype(np.int32)  # (N,)
        yolo_conf = yolo_boxes.conf.cpu().numpy().astype(np.float32)  # (N,)

        # ---- Step 2: SAM2 box-prompt per detection ----
        self._predictor.set_image(image)
        masks_list: list[np.ndarray] = []
        sam2_scores: list[float] = []
        valid_indices: list[int] = []

        for i in range(len(boxes_xyxy)):
            box = boxes_xyxy[i]
            # Skip degenerate boxes.
            if (box[2] - box[0]) < 4 or (box[3] - box[1]) < 4:
                continue
            try:
                mask_out, score_out, _ = self._predictor.predict(
                    box=box,
                    multimask_output=False,
                )
                # mask_out shape: (1, H, W) or (C, H, W)
                mask = mask_out[0].astype(bool) if mask_out.ndim == 3 else mask_out.astype(bool)
                masks_list.append(mask)
                sam2_scores.append(float(score_out[0]) if score_out.ndim == 1 else float(score_out))
                valid_indices.append(i)
            except Exception:
                logger.debug("SAM2 predict failed for box %d, skipping.", i)
                continue

        if not masks_list:
            logger.info("YOLO+SAM2: all %d detections failed SAM2 refinement.", len(boxes_xyxy))
            return {
                "masks": np.zeros((0, h, w), dtype=bool),
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "labels": [],
                "scores": np.zeros(0, dtype=np.float32),
                "risk_levels": [],
                "coco_ids": [],
            }

        masks = np.stack(masks_list, axis=0)  # (M, H, W)
        sam2_scores_arr = np.array(sam2_scores, dtype=np.float32)

        # Filter to valid detections.
        final_boxes = boxes_xyxy[valid_indices]
        final_cls = yolo_cls[valid_indices]
        final_yolo_conf = yolo_conf[valid_indices]

        # Combined score: YOLO confidence × SAM2 quality.
        combined_scores = final_yolo_conf * sam2_scores_arr

        # Build labels and risk levels.
        labels = [_COCO_CLASSES.get(int(c), f"class_{c}") for c in final_cls]
        risk_levels = [_CLASS_RISK.get(lbl, 1) for lbl in labels]
        coco_ids = [int(c) for c in final_cls]

        logger.info(
            "YOLO+SAM2: %d detections → %d refined masks. "
            "Classes: %s",
            len(boxes_xyxy), len(masks), list(set(labels)),
        )
        return {
            "masks": masks,
            "boxes": final_boxes,
            "labels": labels,
            "scores": combined_scores,
            "risk_levels": risk_levels,
            "coco_ids": coco_ids,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _init_yolo(self) -> None:
        """Load YOLO model."""
        try:
            from ultralytics import YOLO

            self._yolo = YOLO(self._yolo_model_name)
            logger.info("YOLO loaded: %s", self._yolo_model_name)
        except ImportError:
            logger.warning(
                "ultralytics not installed — YOLO frontend disabled. "
                "Install with: pip install ultralytics"
            )
            self._use_yolo = False
        except Exception as e:
            logger.warning("YOLO load failed (%s) — YOLO frontend disabled.", e)
            self._use_yolo = False


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

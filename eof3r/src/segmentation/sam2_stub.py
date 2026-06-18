"""SAM2 segmentation stub.

SAM2 could not be cloned from GitHub due to a TLS handshake failure
(gnutls_handshake() failed: The TLS connection was non-properly terminated)
on the current network. This stub provides a synthetic mask generator
so the e2e pipeline can be tested without SAM2.

To enable real SAM2:
  1. Clone https://github.com/facebookresearch/sam2 into baselines/sam2/
  2. Install: pip install -e baselines/sam2/
  3. Replace references of SAM2Stub with SAM2Wrapper in the pipeline.
"""

from __future__ import annotations

import numpy as np


class SAM2Stub:
    """Stub that generates synthetic object masks for testing the pipeline.

    When SAM2 is available, replace this with a real SAM2Wrapper that:
      - Loads SAM2 via build(model_size="base")
      - Calls the automatic mask generator on input images
      - Returns masks, boxes, labels, scores
    """

    def __init__(self, model_size: str = "base", min_mask_area: int = 500) -> None:
        """Initialize stub.

        Args:
            model_size: Ignored in stub. One of tiny/small/base/large.
            min_mask_area: Minimum mask area in pixels (used in synthetic masks).
        """
        self._model_size = model_size
        self._min_mask_area = min_mask_area
        print(f"[SAM2Stub] Using synthetic mask generator (model_size={model_size}).")

    def build(self, model_size: str = "base") -> None:
        """No-op in stub. Real SAM2 loads checkpoint here."""
        self._model_size = model_size
        print("[SAM2Stub] build() called — no-op (stub).")

    def segment(
        self,
        image: np.ndarray,
        box_prompt: bool = False,
    ) -> dict:
        """Generate synthetic segmentation masks.

        Args:
            image: RGB image (H, W, 3), uint8 [0, 255].
            box_prompt: If True, use box-prompt mode (stub ignores this).

        Returns:
            Dict with keys:
              - masks: (N, H, W) binary masks, one per detected object.
              - boxes: (N, 4) bounding boxes, xyxy format.
              - labels: list of N class-label strings (placeholder).
              - scores: (N,) confidence scores.
        """
        h, w = image.shape[:2]
        # Generate 2-4 synthetic rectangular masks.
        num_objs = np.random.randint(2, 5)
        masks = np.zeros((num_objs, h, w), dtype=bool)
        boxes = np.zeros((num_objs, 4), dtype=np.float32)
        labels: list[str] = []
        scores = np.ones(num_objs, dtype=np.float32) * 0.9

        class_names = ["box", "cone", "person", "bicycle"]

        for i in range(num_objs):
            # Random rectangle covering 10-30% of the image.
            bw = np.random.randint(w // 6, w // 3)
            bh = np.random.randint(h // 6, h // 3)
            x1 = np.random.randint(0, w - bw)
            y1 = np.random.randint(0, h - bh)
            x2, y2 = x1 + bw, y1 + bh
            masks[i, y1:y2, x1:x2] = True
            boxes[i] = [x1, y1, x2, y2]
            labels.append(class_names[i % len(class_names)])

        print(f"[SAM2Stub] Generated {num_objs} synthetic masks on image {w}x{h}.")
        return {
            "masks": masks,
            "boxes": boxes,
            "labels": labels,
            "scores": scores,
        }

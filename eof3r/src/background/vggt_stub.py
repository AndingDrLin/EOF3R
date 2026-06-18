"""VGGT background estimation stub.

VGGT could not be cloned from GitHub due to a TLS handshake failure
(gnutls_handshake() failed: The TLS connection was non-properly terminated)
on the current network. This stub provides synthetic geometry output
so the e2e pipeline can be tested without VGGT.

To enable real VGGT:
  1. Clone https://github.com/facebookresearch/vggt into baselines/vggt/
  2. Install its dependencies (see baselines/vggt/requirements.txt)
  3. Download weights and place in checkpoints/
  4. Replace references of VGGTStub with VGGTWrapper in the pipeline.

Expected VGGTWrapper API:
  class VGGTWrapper:
      def build(self, checkpoint_path: str | None = None) -> None: ...
      def infer(self, images: list[np.ndarray]) -> dict:
          # Returns: pointmap (N,H,W,3), camera_poses (N,4,4),
          #          ground_plane (4,), drivable_mask (H,W)
"""

from __future__ import annotations

import numpy as np


class VGGTStub:
    """Stub that generates synthetic background geometry for testing.

    When VGGT is available, replace this with real VGGTWrapper.
    """

    def __init__(
        self,
        max_resolution: int = 512,
        estimate_ground: bool = True,
        estimate_drivable: bool = True,
    ) -> None:
        """Initialize stub.

        Args:
            max_resolution: Max input resolution (ignored in stub).
            estimate_ground: Whether to estimate ground plane (stub returns flat plane).
            estimate_drivable: Whether to estimate drivable region (stub returns floor).
        """
        self._max_resolution = max_resolution
        self._estimate_ground = estimate_ground
        self._estimate_drivable = estimate_drivable
        print("[VGGTStub] Using synthetic geometry generator.")

    def build(self, checkpoint_path: str | None = None) -> None:
        """No-op in stub. Real VGGT loads checkpoint here."""
        print(f"[VGGTStub] build(checkpoint_path={checkpoint_path}) — no-op (stub).")

    def infer(self, images: list[np.ndarray]) -> dict:
        """Generate synthetic background geometry.

        Args:
            images: List of RGB images (H, W, 3), uint8 [0, 255].

        Returns:
            Dict with keys:
              - pointmap: (N, H/4, W/4, 3) float32 — 3D points in Y-up coords.
              - camera_poses: (N, 4, 4) float32 — world-from-camera matrices.
              - ground_plane: (4,) float32 — (a, b, c, d) for ax+by+cz+d=0 in Y-up.
              - drivable_mask: (H/4, W/4) bool — estimated traversable floor region.
        """
        n = len(images)
        first_h, first_w = images[0].shape[:2]
        # Downsampled pointmap resolution.
        h_out, w_out = first_h // 4, first_w // 4

        pointmaps = []
        camera_poses = []

        # Simulate cameras spaced along X-axis looking at origin.
        for i in range(n):
            # Pointmap: random scatter around origin, Y-up.
            # Real VGGT would produce dense, structured output.
            pm = np.random.randn(h_out, w_out, 3).astype(np.float32) * 2.0
            # Flatten Y (height) to simulate ground plane at y=0.
            pm[:, :, 1] = np.abs(pm[:, :, 1]) * 0.5
            # Push majority below horizon down to near-ground.
            pm[h_out // 2 :, :, 1] = np.random.rand(h_out - h_out // 2, w_out).astype(np.float32) * 0.3
            pointmaps.append(pm)

            # Camera pose: identity-like with slight offset.
            pose = np.eye(4, dtype=np.float32)
            pose[0, 3] = 0.5 * i  # cameras spaced along X
            pose[1, 3] = 1.5  # camera height = 1.5m
            pose[2, 3] = 3.0  # camera facing +Z (forward in Y-up)
            camera_poses.append(pose)

        # Ground plane: y=0 in Y-up (horizontal).
        ground_plane = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        # Drivable mask: lower half of the image (floor region).
        drivable_mask = np.zeros((h_out, w_out), dtype=bool)
        drivable_mask[h_out // 2 :, :] = True

        print(
            f"[VGGTStub] Generated synthetic pointmaps ({n}, {h_out}x{w_out}x3), "
            f"camera_poses ({n} views), ground_plane, drivable_mask."
        )
        return {
            "pointmap": np.stack(pointmaps, axis=0).astype(np.float32),
            "camera_poses": np.stack(camera_poses, axis=0).astype(np.float32),
            "ground_plane": ground_plane,
            "drivable_mask": drivable_mask,
        }

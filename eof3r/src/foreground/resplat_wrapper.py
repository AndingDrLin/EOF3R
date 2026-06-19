"""ReSplat wrapper for EOF3R Phase B.

Wraps the ReSplat (Recurrent Gaussian Splatting) model behind the same
API as MVSplatWrapper: build() → infer() → GaussianData.

ReSplat key properties:
    - 16× fewer Gaussians than per-pixel methods (~8K vs 131K)
    - Recurrent refinement via rendering error feedback
    - Uses gsplat renderer (not original 3DGS CUDA)
    - MIT license

The wrapper isolates ReSplat's dependencies (Python 3.12, PyTorch 2.7.0,
gsplat) from the main eof3r environment via sys.path manipulation.

Usage:
    resplat = ReSplatWrapper()
    resplat.build(checkpoint="haofeixu/resplat-base-re10k")
    output = resplat.infer(images, poses, K)
    gaussians = output["gaussians"]  # GaussianData
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Gaussian data structure (compatible with MVSplatWrapper's GaussianData)
from eof3r.src.foreground.mvsplat_wrapper import GaussianData


class ReSplatWrapper:
    """Wrapper for the ReSplat feedforward 3DGS model.

    Follows the same API as MVSplatWrapper:
        build() → infer() → {"gaussians": GaussianData, "metadata": dict}

    The wrapper handles:
        1. Path isolation (ReSplat has its own Python/PyTorch version)
        2. Checkpoint loading (from HuggingFace or local path)
        3. Gaussian parameter extraction and format conversion
        4. Integration with occupancy/semantic heads
    """

    def __init__(self):
        self.model = None
        self.config = None
        self._resplat_root = None
        self._original_sys_path = None

    def build(
        self,
        checkpoint: str = "haofeixu/resplat-base-re10k",
        resplat_root: Optional[str] = None,
        device: str = "cuda",
        num_refine: int = 0,
    ) -> None:
        """Load ReSplat model.

        Args:
            checkpoint: HuggingFace model ID or local path to checkpoint.
            resplat_root: Path to ReSplat repo. If None, uses baselines/resplat/.
            device: Device for inference.
            num_refine: Number of recurrent refinement iterations (0 = no refinement).
        """
        # Resolve ReSplat root
        if resplat_root is not None:
            self._resplat_root = Path(resplat_root)
        else:
            # Try environment variable, then baselines/
            import os
            env_root = os.environ.get("RESPLAT_ROOT")
            if env_root:
                self._resplat_root = Path(env_root)
            else:
                self._resplat_root = Path(__file__).parents[3] / "baselines" / "resplat"

        if not self._resplat_root.exists():
            raise FileNotFoundError(
                f"ReSplat not found at {self._resplat_root}\n"
                f"Clone: git clone https://github.com/cvg/ReSplat baselines/resplat"
            )

        # Path isolation: temporarily add ReSplat to sys.path
        self._enter_resplat_env()

        try:
            self._load_model(checkpoint, device, num_refine)
        finally:
            self._exit_resplat_env()

        logger.info(f"ReSplat loaded: checkpoint={checkpoint}, device={device}")

    def _enter_resplat_env(self) -> None:
        """Temporarily modify sys.path for ReSplat imports."""
        self._original_sys_path = sys.path.copy()

        # Add ReSplat source to path
        resplat_src = self._resplat_root / "src"
        if str(resplat_src) not in sys.path:
            sys.path.insert(0, str(resplat_src))

    def _exit_resplat_env(self) -> None:
        """Restore original sys.path."""
        if self._original_sys_path is not None:
            sys.path = self._original_sys_path
            self._original_sys_path = None

    def _load_model(
        self,
        checkpoint: str,
        device: str,
        num_refine: int,
    ) -> None:
        """Load ReSplat model from checkpoint.

        This method runs within the ReSplat environment (sys.path modified).
        """
        from omegaconf import OmegaConf
        from src.config import RootCfg
        from src.model.model_wrapper import ModelWrapper

        # Load config
        config_path = self._resplat_root / "config" / "main.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"ReSplat config not found: {config_path}")

        # Use Hydra-style config loading
        from hydra import compose, initialize_config_dir

        config_dir = str(self._resplat_root / "config")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="main")

        # Override num_refine if specified
        if num_refine > 0:
            cfg.model.encoder.num_refine = num_refine

        # Convert to typed config
        root_cfg = OmegaConf.structured(RootCfg)
        root_cfg = OmegaConf.merge(root_cfg, cfg)

        # Load checkpoint
        if checkpoint.startswith("haofeixu/") or checkpoint.startswith("hf://"):
            # HuggingFace model ID
            from huggingface_hub import hf_hub_download

            model_id = checkpoint.replace("hf://", "")
            checkpoint_path = hf_hub_download(
                repo_id=model_id,
                filename="model.ckpt",
            )
        else:
            checkpoint_path = checkpoint

        # Build model
        self.model = ModelWrapper.load_from_checkpoint(
            checkpoint_path,
            cfg=root_cfg.model,
            loss_fns=[],
            test_cfg=root_cfg.test,
        )
        self.model = self.model.to(device)
        self.model.eval()

        self.config = root_cfg
        self.device = device

    @torch.no_grad()
    def infer(
        self,
        images: np.ndarray | torch.Tensor,
        poses: np.ndarray | torch.Tensor,
        K: np.ndarray | torch.Tensor,
    ) -> dict[str, Any]:
        """Run ReSplat inference on input views.

        Args:
            images: (B, V, 3, H, W) input images in [0, 1].
            poses: (B, V, 4, 4) camera poses (world-from-camera, OpenCV convention).
            K: (B, V, 3, 3) camera intrinsics.

        Returns:
            Dict with:
                - "gaussians": GaussianData with means, opacities, scales, etc.
                - "metadata": dict with additional info (depth, etc.)
        """
        if self.model is None:
            raise RuntimeError("Call build() before infer()")

        # Convert to tensors if needed
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float()
        if isinstance(poses, np.ndarray):
            poses = torch.from_numpy(poses).float()
        if isinstance(K, np.ndarray):
            K = torch.from_numpy(K).float()

        # Ensure batch dimension
        if images.ndim == 4:
            images = images.unsqueeze(0)
        if poses.ndim == 3:
            poses = poses.unsqueeze(0)
        if K.ndim == 2:
            K = K.unsqueeze(0)

        B, V, C, H, W = images.shape
        device = self.device

        images = images.to(device)
        poses = poses.to(device)
        K = K.to(device)

        # Enter ReSplat env for model forward
        self._enter_resplat_env()
        try:
            # Build context batch in ReSplat format
            context = {
                "image": images,
                "extrinsics": poses,
                "intrinsics": K,
                "near": torch.full((B, V), 0.1, device=device),
                "far": torch.full((B, V), 100.0, device=device),
                "index": torch.arange(V, device=device).unsqueeze(0).expand(B, -1),
            }

            # Encoder forward pass
            encoder_output = self.model.encoder(context, 0, deterministic=True)
            gaussians = encoder_output["gaussians"]
            depths = encoder_output.get("depths", None)

        finally:
            self._exit_resplat_env()

        # Extract Gaussian parameters
        means = gaussians.means              # (B, G, 3)
        opacities = gaussians.opacities      # (B, G)
        scales = gaussians.scales            # (B, G, 3)
        rotations = gaussians.rotations      # (B, G, 4)
        harmonics = gaussians.harmonics      # (B, G, 3, d_sh)
        covariances = gaussians.covariances  # (B, G, 3, 3)

        # Convert to GaussianData (numpy, CPU)
        gaussians_data = GaussianData(
            means=means[0].cpu().numpy(),           # (G, 3)
            opacities=opacities[0].cpu().numpy(),   # (G,)
            scales=scales[0].cpu().numpy(),         # (G, 3)
            covariances=covariances[0].cpu().numpy(),  # (G, 3, 3)
            harmonics=harmonics[0].cpu().numpy().transpose(0, 2, 1),  # (G, d_sh, 3) from (G, 3, d_sh)
            rotations=rotations[0].cpu().numpy(),   # (G, 4)
        )

        # Metadata
        metadata = {
            "num_gaussians": gaussians_data.means.shape[0],
            "model": "resplat",
            "device": str(device),
        }

        if depths is not None:
            metadata["depths"] = depths[0].cpu().numpy()

        return {
            "gaussians": gaussians_data,
            "metadata": metadata,
        }

    def extract_occupancy(
        self,
        gaussians: GaussianData,
        config: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Extract occupancy data from Gaussians for BEV projection.

        Compatible with MVSplatWrapper.extract_occupancy().

        Args:
            gaussians: GaussianData from infer().
            config: Optional configuration overrides.

        Returns:
            Dict with occupancy-relevant fields.
        """
        config = config or {}
        opacity_threshold = config.get("opacity_threshold", 0.5)
        height_range = config.get("height_range", (0.0, 3.0))

        means = gaussians.means
        opacities = gaussians.opacities
        scales = gaussians.scales

        # Height filtering
        height_mask = (means[:, 1] >= height_range[0]) & (means[:, 1] <= height_range[1])

        # Opacity filtering
        opacity_mask = opacities > opacity_threshold

        # Combined mask
        valid_mask = height_mask & opacity_mask

        return {
            "means": means[valid_mask],
            "opacities": opacities[valid_mask],
            "scales": scales[valid_mask],
            "num_total": len(means),
            "num_valid": int(valid_mask.sum()),
            "opacity_mean": float(opacities.mean()),
            "opacity_std": float(opacities.std()),
        }

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        if self.model is None:
            raise RuntimeError("No model to save")
        torch.save({
            "model_state": self.model.state_dict(),
            "config": self.config,
        }, path)

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.config = checkpoint["config"]

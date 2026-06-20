"""Phase B training script: Cross-model geometric distillation.

Trains ReSplat with VGGT geometric supervision using the three-stage
schedule from Phase B design §2.5.

Prerequisites:
    1. Pre-compute VGGT supervision:
       python eof3r/scripts/preprocess/precompute_vggt_supervision.py

    2. Ensure ReSplat is available:
       git clone https://github.com/cvg/ReSplat baselines/resplat

    3. Environment: conda activate eof3r (for now; later may need resplat env)

Usage:
    # Train with default config
    python eof3r/scripts/train/train_phase_b.py

    # Train with custom config
    python eof3r/scripts/train/train_phase_b.py --config eof3r/configs/phase_b.yaml

    # Resume from checkpoint
    python eof3r/scripts/train/train_phase_b.py --resume outputs/phase_b/checkpoint_step_005000.pt

    # Debug mode (small dataset, fast)
    python eof3r/scripts/train/train_phase_b.py --debug
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from eof3r.src.training.heads import OccupancyHead, SemanticHead
from eof3r.src.training.trainer import PhaseBTrainer, PhaseBConfig


class MockGaussianParams:
    """Mock Gaussian parameters matching ReSplat's output format."""
    def __init__(self, means, scales, quats, opacities):
        self.means = means        # (B, G, 3)
        self.scales = scales      # (B, G, 3)
        self.quats = quats        # (B, G, 4)
        self.opacities = opacities  # (B, G)

    @property
    def covariances(self):
        """Compute covariance from scales and rotations (simplified: diagonal)."""
        # (B, G, 3, 3) diagonal covariance from scales
        B, G, _ = self.scales.shape
        cov = torch.zeros(B, G, 3, 3, device=self.scales.device)
        cov[:, :, 0, 0] = self.scales[:, :, 0] ** 2
        cov[:, :, 1, 1] = self.scales[:, :, 1] ** 2
        cov[:, :, 2, 2] = self.scales[:, :, 2] ** 2
        return cov


class MockEncoder(nn.Module):
    """Mock encoder that produces Gaussian parameters from images.

    Replaces ReSplat encoder for testing the training pipeline without
    the actual model. Produces learnable Gaussian parameters so the
    training loop can run end-to-end.
    """
    def __init__(self, num_gaussians: int = 4096):
        super().__init__()
        self.num_gaussians = num_gaussians
        # Learnable Gaussian parameters (will be shaped by training)
        self._means = nn.Parameter(torch.randn(1, num_gaussians, 3) * 0.5)
        self._scales = nn.Parameter(torch.ones(1, num_gaussians, 3) * 0.01)
        self._quats = nn.Parameter(
            torch.tensor([1.0, 0.0, 0.0, 0.0]).unsqueeze(0).unsqueeze(0).expand(1, num_gaussians, 4).clone()
        )
        self._opacities = nn.Parameter(torch.zeros(1, num_gaussians))

    def forward(self, context, step=0, deterministic=False):
        B = context["image"].shape[0]
        means = self._means.expand(B, -1, -1)
        scales = self._scales.expand(B, -1, -1).abs()  # Ensure positive
        quats = self._quats.expand(B, -1, -1)
        quats = quats / quats.norm(dim=-1, keepdim=True)  # Normalize quaternion
        opacities = torch.sigmoid(self._opacities.expand(B, -1))

        gaussians = MockGaussianParams(means, scales, quats, opacities)
        return {"gaussians": gaussians}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_resplat_encoder(config: PhaseBConfig):
    """Load ReSplat encoder for training.

    Uses the lightweight loader that bypasses ReSplat's Hydra config system.
    Requires the resplat conda env (Python 3.12 + PyTorch 2.7 + gsplat + pointops).
    """
    import os
    checkpoint_path = os.environ.get(
        "RESPLAT_CHECKPOINT",
        str(PROJECT_ROOT / "baselines" / "resplat" / "pretrained"
            / "resplat-base-re10k-256x256-view2-b90d1b53.pth"),
    )
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"ReSplat checkpoint not found: {checkpoint_path}\n"
            f"Download from MODEL_ZOO.md or set RESPLAT_CHECKPOINT env var."
        )

    from eof3r.src.foreground.resplat_encoder import load_resplat_encoder as _load
    return _load(checkpoint_path)


def create_heads(config: PhaseBConfig):
    """Create occupancy and semantic heads."""
    occ_head = OccupancyHead(
        input_dim=10,  # 3 means + 3 scales + 4 rotation
        hidden_dims=config.occ_hidden_dims,
        dropout=0.1,
        use_visual_features=True,  # Fix 2: gated fusion with visual features
        visual_feature_dim=4,  # RGB + depth from image projection
    )

    sem_head = SemanticHead(
        num_classes=config.num_classes,
        input_dim=10,
        hidden_dims=config.sem_hidden_dims,
        dropout=0.1,
    )

    return occ_head, sem_head


def create_dataloaders(config: PhaseBConfig, supervision_dir: str | None = None):
    """Create train and validation dataloaders from pre-computed supervision."""
    from torch.utils.data import DataLoader
    from eof3r.src.training.trainer import VGGTSupervisionDataset

    if supervision_dir is None:
        import os
        data_root = Path(os.environ.get("EOF3R_DATA", "/data/EOF3R"))
        supervision_dir = data_root / "processed" / "vggt_supervision"
    else:
        supervision_dir = Path(supervision_dir)

    if not supervision_dir.exists():
        raise FileNotFoundError(
            f"VGGT supervision not found at {supervision_dir}\n"
            f"Run: python eof3r/scripts/preprocess/precompute_vggt_supervision.py"
        )

    # The directory itself may contain manifest.txt (split-specific) or have train/test subdirs
    train_dataset = VGGTSupervisionDataset(supervision_dir, split="train")
    val_dataset = VGGTSupervisionDataset(supervision_dir, split="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def create_mock_dataloaders(config: PhaseBConfig):
    """Create mock dataloaders for debugging without real data."""
    from torch.utils.data import Dataset, DataLoader

    class MockDataset(Dataset):
        def __init__(self, size: int = 100):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            V, H, W = 2, 256, 256
            G = 4096  # ~4K Gaussians

            return {
                "images": torch.randn(V, 3, H, W),
                "poses": torch.eye(4).unsqueeze(0).expand(V, -1, -1),
                "intrinsics": torch.eye(3).unsqueeze(0).expand(V, -1, -1),
                "depth": torch.rand(V, H, W) * 10,
                "surface_points": torch.randn(1000, 3),
                "labels": {
                    "occupied": torch.randint(0, 2, (G,)).bool(),
                    "free": torch.randint(0, 2, (G,)).bool(),
                },
            }

    train_dataset = MockDataset(size=100)
    val_dataset = MockDataset(size=20)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(
        description="Phase B training: Cross-model geometric distillation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to Phase B config YAML (default: use PhaseBConfig defaults)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: small dataset, fast iteration",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for training",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/phase_b",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="Override total training steps",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--lr-encoder",
        type=float,
        default=None,
        help="Override encoder learning rate",
    )
    parser.add_argument(
        "--lr-heads",
        type=float,
        default=None,
        help="Override heads learning rate",
    )
    parser.add_argument(
        "--alpha", type=float, default=None, help="Override L_depth weight (Chamfer)"
    )
    parser.add_argument(
        "--beta", type=float, default=None, help="Override L_occ weight (Focal)"
    )
    parser.add_argument(
        "--gamma", type=float, default=None, help="Override L_free weight (Hinge)"
    )
    parser.add_argument(
        "--delta", type=float, default=None, help="Override L_sem weight (CE)"
    )
    parser.add_argument(
        "--eta", type=float, default=None, help="Override L_color weight (auxiliary)"
    )
    parser.add_argument(
        "--theta", type=float, default=None,
        help="Override L_position weight (occupancy-guided position loss)",
    )
    parser.add_argument(
        "--zeta", type=float, default=None,
        help="Override L_bev weight (BEV coverage loss)",
    )
    parser.add_argument(
        "--uniform-weights",
        action="store_true",
        help="Use uniform loss weights across all stages (no stage schedule)",
    )
    parser.add_argument(
        "--use-bce",
        action="store_true",
        help="Use BCE instead of focal loss for occupancy (ablation)",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=None,
        help="Override kappa (adaptive threshold multiplier for Gaussian-surface labeling)",
    )
    parser.add_argument(
        "--supervision-dir",
        type=str,
        default=None,
        help="Path to pre-computed VGGT supervision directory (contains manifest.txt)",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze encoder, only train occupancy/semantic heads",
    )
    args = parser.parse_args()

    # Build config
    config = PhaseBConfig()

    if args.config:
        # Load from YAML (if implemented)
        logger.info(f"Loading config from {args.config}")
        # TODO: YAML config loading
        pass

    # Apply CLI overrides
    if args.debug:
        config.total_steps = 1000
        config.batch_size = 1
        config.log_interval = 10
        config.eval_interval = 100
        config.save_interval = 500
        logger.info("Debug mode: reduced steps and batch size")

    if args.resume:
        config.resume_from = args.resume

    if args.output_dir:
        config.output_dir = args.output_dir

    if args.total_steps:
        config.total_steps = args.total_steps

    if args.batch_size:
        config.batch_size = args.batch_size

    if args.lr_encoder:
        config.lr_encoder = args.lr_encoder

    if args.lr_heads:
        config.lr_heads = args.lr_heads

    # Apply loss weight overrides
    loss_overrides = {}
    if args.alpha is not None:
        loss_overrides["alpha"] = args.alpha
    if args.beta is not None:
        loss_overrides["beta"] = args.beta
    if args.gamma is not None:
        loss_overrides["gamma"] = args.gamma
    if args.delta is not None:
        loss_overrides["delta"] = args.delta
    if args.eta is not None:
        loss_overrides["eta"] = args.eta
    if args.theta is not None:
        loss_overrides["theta"] = args.theta
    if args.zeta is not None:
        loss_overrides["zeta"] = args.zeta

    if args.use_bce:
        config.use_bce = True
        logger.info("Using BCE instead of focal loss for occupancy")

    if args.kappa is not None:
        config.kappa = args.kappa
        logger.info(f"Kappa override: {args.kappa}")

    if loss_overrides or args.uniform_weights:
        if args.uniform_weights:
            # Use same weights for all stages (ablation: no schedule)
            base_weights = {
                "alpha": args.alpha if args.alpha is not None else 1.0,
                "beta": args.beta if args.beta is not None else 1.0,
                "gamma": args.gamma if args.gamma is not None else 1.0,
                "delta": args.delta if args.delta is not None else 0.3,
                "eta": args.eta if args.eta is not None else 0.1,
                "theta": args.theta if args.theta is not None else 0.5,
                "zeta": args.zeta if args.zeta is not None else 0.3,
            }
            config.stage_weights = {1: base_weights, 2: base_weights, 3: base_weights}
            logger.info(f"Uniform weights across all stages: {base_weights}")
        else:
            # Override specific weights in all stages
            for stage in config.stage_weights:
                for k, v in loss_overrides.items():
                    config.stage_weights[stage][k] = v
            logger.info(f"Loss weight overrides applied: {loss_overrides}")

    # Log config
    logger.info("=" * 60)
    logger.info("Phase B Training Configuration")
    logger.info(f"Total steps: {config.total_steps}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"LR encoder: {config.lr_encoder}")
    logger.info(f"LR heads: {config.lr_heads}")
    logger.info(f"Output dir: {config.output_dir}")
    logger.info("=" * 60)

    # Create model components
    logger.info("Loading ReSplat encoder...")
    try:
        encoder = load_resplat_encoder(config)
    except Exception as e:
        logger.warning(f"Failed to load ReSplat: {e}")
        logger.info("Using mock encoder for debugging")
        encoder = MockEncoder(num_gaussians=4096)

    logger.info("Creating occupancy and semantic heads...")
    occ_head, sem_head = create_heads(config)

    # Create dataloaders
    logger.info("Creating dataloaders...")
    try:
        train_loader, val_loader = create_dataloaders(config, args.supervision_dir)
    except FileNotFoundError as e:
        logger.warning(f"Pre-computed supervision not found: {e}")
        logger.info("Using mock dataloaders for debugging")
        train_loader, val_loader = create_mock_dataloaders(config)

    # Create trainer
    trainer = PhaseBTrainer(
        resplat_encoder=encoder,
        occupancy_head=occ_head,
        semantic_head=sem_head,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=args.device,
    )

    # Freeze encoder if requested
    if args.freeze_encoder:
        for param in trainer.encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder frozen — only training heads")

    # Start training
    logger.info("Starting Phase B training...")
    trainer.train()

    logger.info("Training complete!")
    logger.info(f"Checkpoints saved to: {config.output_dir}")


if __name__ == "__main__":
    main()

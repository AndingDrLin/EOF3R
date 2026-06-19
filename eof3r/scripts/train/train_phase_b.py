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

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from eof3r.src.training.heads import OccupancyHead, SemanticHead
from eof3r.src.training.trainer import PhaseBTrainer, PhaseBConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_resplat_encoder(config: PhaseBConfig):
    """Load ReSplat encoder for training.

    The encoder is the pretrained ReSplat model. During Phase B training,
    we fine-tune it with geometric supervision while training the new
    occupancy and semantic heads.
    """
    from eof3r.src.foreground.resplat_wrapper import ReSplatWrapper

    resplat = ReSplatWrapper()
    resplat.build(
        checkpoint="haofeixu/resplat-base-re10k",
        num_refine=0,  # No refinement during training
    )

    return resplat.model.encoder


def create_heads(config: PhaseBConfig):
    """Create occupancy and semantic heads."""
    occ_head = OccupancyHead(
        input_dim=10,  # 3 means + 3 scales + 4 rotation
        hidden_dims=config.occ_hidden_dims,
        dropout=0.1,
    )

    sem_head = SemanticHead(
        num_classes=config.num_classes,
        input_dim=10,
        hidden_dims=config.sem_hidden_dims,
        dropout=0.1,
    )

    return occ_head, sem_head


def create_dataloaders(config: PhaseBConfig):
    """Create train and validation dataloaders from pre-computed supervision."""
    from torch.utils.data import DataLoader
    from eof3r.src.training.trainer import VGGTSupervisionDataset

    import os
    data_root = Path(os.environ.get("EOF3R_DATA", "/data/EOF3R"))
    supervision_dir = data_root / "processed" / "vggt_supervision"

    if not supervision_dir.exists():
        raise FileNotFoundError(
            f"VGGT supervision not found at {supervision_dir}\n"
            f"Run: python eof3r/scripts/preprocess/precompute_vggt_supervision.py"
        )

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
        encoder = torch.nn.Linear(10, 10)  # Placeholder

    logger.info("Creating occupancy and semantic heads...")
    occ_head, sem_head = create_heads(config)

    # Create dataloaders
    logger.info("Creating dataloaders...")
    try:
        train_loader, val_loader = create_dataloaders(config)
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

    # Start training
    logger.info("Starting Phase B training...")
    trainer.train()

    logger.info("Training complete!")
    logger.info(f"Checkpoints saved to: {config.output_dir}")


if __name__ == "__main__":
    main()

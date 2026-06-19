"""Three-stage training loop for Phase B cross-model geometric distillation.

Implements the training schedule from Phase B design §2.5:
    Stage 1 (Warmup, ~30%): Move Gaussians to VGGT surfaces
    Stage 2 (Main, ~50%): Train occupancy + free-space + semantics
    Stage 3 (Fine, ~20%): Refine, color signal exits

The trainer orchestrates:
    1. ReSplat encoder forward → Gaussian parameters
    2. Occupancy/Semantic head predictions
    3. VGGT supervision labeling (pre-computed or online)
    4. Loss computation with stage-dependent weights
    5. Gradient update with stage-dependent learning rates

Usage:
    trainer = PhaseBTrainer(
        resplat_encoder=encoder,
        occupancy_head=occ_head,
        semantic_head=sem_head,
        vggt_supervision_dir="/path/to/precomputed/",
        config=config,
    )
    trainer.train()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from eof3r.src.training.losses import compute_total_loss, get_stage_weights
from eof3r.src.training.heads import OccupancyHead, SemanticHead
from eof3r.src.training.supervision import (
    GaussianLabels,
    label_gaussians_by_vggt_projection,
    compute_vggt_surface_points,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PhaseBConfig:
    """Configuration for Phase B training.

    Follows CLAUDE.md §6: all hyperparameters in config, no code defaults.
    """
    # Training schedule
    total_steps: int = 100_000
    stage1_ratio: float = 0.3  # Warmup fraction
    stage2_ratio: float = 0.5  # Main training fraction
    stage3_ratio: float = 0.2  # Fine-tune fraction

    # Optimizer
    lr_encoder: float = 1e-4  # Lower LR for pretrained encoder
    lr_heads: float = 1e-3   # Higher LR for new heads
    weight_decay: float = 1e-4
    gradient_clip_val: float = 0.5

    # Loss hyperparameters
    focal_gamma: float = 2.0
    hinge_epsilon: float = 0.05
    ssim_weight: float = 0.2
    label_smoothing: float = 0.1
    kappa: float = 3.0  # Adaptive threshold multiplier

    # Data
    batch_size: int = 1
    num_workers: int = 4
    image_shape: tuple[int, int] = (256, 256)

    # Logging
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    output_dir: str = "outputs/phase_b"

    # Checkpointing
    resume_from: Optional[str] = None
    save_best: bool = True

    # Semantic classes (COCO subset for campus delivery)
    num_classes: int = 10  # e.g., person, bicycle, car, truck, ...

    # Occupancy head architecture
    occ_hidden_dims: list[int] = field(default_factory=lambda: [64, 32])
    sem_hidden_dims: list[int] = field(default_factory=lambda: [64, 32])

    # Loss weights per stage (from Phase B design §2.5)
    stage_weights: dict[int, dict[str, float]] = field(default_factory=lambda: {
        1: {"alpha": 1.0, "beta": 0.3, "gamma": 0.1, "delta": 0.0, "eta": 0.3},
        2: {"alpha": 0.5, "beta": 1.0, "gamma": 0.5, "delta": 0.3, "eta": 0.1},
        3: {"alpha": 0.3, "beta": 1.0, "gamma": 1.0, "delta": 0.5, "eta": 0.05},
    })


# ---------------------------------------------------------------------------
# Training State
# ---------------------------------------------------------------------------

@dataclass
class TrainingState:
    """Tracks training progress."""
    global_step: int = 0
    epoch: int = 0
    best_metric: float = float("inf")
    stage: int = 1
    losses_history: list[dict[str, float]] = field(default_factory=list)
    start_time: float = 0.0

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time

    @property
    def stage_progress(self) -> float:
        """Progress within current stage (0 to 1)."""
        # This is approximate; exact depends on stage boundaries
        return min(1.0, self.global_step / max(1, 100_000))


# ---------------------------------------------------------------------------
# Pre-computed Supervision Dataset
# ---------------------------------------------------------------------------

class VGGTSupervisionDataset:
    """Dataset of pre-computed VGGT supervision for ReSplat training.

    Pre-computed by eof3r/scripts/preprocess/precompute_vggt_supervision.py.
    Each sample contains:
        - images: input RGB images
        - camera poses and intrinsics
        - VGGT depth maps
        - VGGT surface point clouds
        - Per-Gaussian labels (occupied/free/unknown) [computed online or pre-computed]
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.samples = self._load_manifest()

    def _load_manifest(self) -> list[dict[str, Path]]:
        """Load list of pre-computed supervision files."""
        manifest_path = self.data_dir / self.split / "manifest.txt"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Supervision manifest not found: {manifest_path}\n"
                f"Run: python eof3r/scripts/preprocess/precompute_vggt_supervision.py"
            )

        samples = []
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    sample_dir = self.data_dir / self.split / line
                    samples.append({
                        "images": sample_dir / "images.pt",
                        "depth": sample_dir / "depth.pt",
                        "poses": sample_dir / "poses.pt",
                        "intrinsics": sample_dir / "intrinsics.pt",
                        "points": sample_dir / "surface_points.pt",
                        "labels": sample_dir / "labels.pt",
                    })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Load a pre-computed supervision sample."""
        paths = self.samples[idx]
        return {
            "images": torch.load(paths["images"], weights_only=True),
            "depth": torch.load(paths["depth"], weights_only=True),
            "poses": torch.load(paths["poses"], weights_only=True),
            "intrinsics": torch.load(paths["intrinsics"], weights_only=True),
            "surface_points": torch.load(paths["points"], weights_only=True),
            "labels": torch.load(paths["labels"], weights_only=True),
        }


# ---------------------------------------------------------------------------
# Phase B Trainer
# ---------------------------------------------------------------------------

class PhaseBTrainer:
    """Three-stage trainer for cross-model geometric distillation.

    Orchestrates ReSplat encoder + occupancy/semantic heads training
    with VGGT geometric supervision.
    """

    def __init__(
        self,
        # Model components (injected, not created here)
        resplat_encoder: nn.Module,
        occupancy_head: OccupancyHead,
        semantic_head: SemanticHead,
        # Data
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        # Config
        config: Optional[PhaseBConfig] = None,
        # Device
        device: str = "cuda",
    ):
        self.config = config or PhaseBConfig()
        self.device = torch.device(device)

        # Models
        self.encoder = resplat_encoder.to(self.device)
        self.occ_head = occupancy_head.to(self.device)
        self.sem_head = semantic_head.to(self.device)

        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optimizer with differential LR
        self.optimizer = torch.optim.AdamW([
            {"params": self.encoder.parameters(), "lr": self.config.lr_encoder},
            {"params": self.occ_head.parameters(), "lr": self.config.lr_heads},
            {"params": self.sem_head.parameters(), "lr": self.config.lr_heads},
        ], weight_decay=self.config.weight_decay)

        # LR scheduler: cosine annealing with warmup
        # OneCycleLR requires total_steps > 1/pct_start
        # For small total_steps (debug), use CosineAnnealingLR instead
        if self.config.total_steps > 20:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=[self.config.lr_encoder, self.config.lr_heads, self.config.lr_heads],
                total_steps=self.config.total_steps,
                pct_start=0.05,  # 5% warmup
                anneal_strategy="cos",
            )
        else:
            # Fallback for debug mode
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.total_steps,
            )

        # State
        self.state = TrainingState(start_time=time.time())

        # Output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure logging to file and stdout."""
        log_file = self.output_dir / "training.log"
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s"
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    def train(self) -> None:
        """Main training loop."""
        logger.info("=" * 60)
        logger.info("Phase B Training: Cross-Model Geometric Distillation")
        logger.info(f"Total steps: {self.config.total_steps}")
        logger.info(f"Device: {self.device}")
        logger.info("=" * 60)

        # Resume if specified
        if self.config.resume_from:
            self._load_checkpoint(self.config.resume_from)

        # Training loop
        while self.state.global_step < self.config.total_steps:
            self.state.epoch += 1
            for batch in self.train_loader:
                if self.state.global_step >= self.config.total_steps:
                    break

                # Determine current stage
                self.state.stage = self._get_current_stage()

                # Get stage-specific loss weights
                weights = get_stage_weights(
                    self.state.global_step,
                    self.config.total_steps,
                    self.config.stage_weights,
                )

                # Training step
                losses = self._train_step(batch, weights)

                # Update learning rate
                self.scheduler.step()

                # Logging
                if self.state.global_step % self.config.log_interval == 0:
                    self._log_losses(losses, weights)

                # Evaluation
                if (self.val_loader is not None and
                    self.state.global_step % self.config.eval_interval == 0):
                    self._evaluate()

                # Checkpointing
                if self.state.global_step % self.config.save_interval == 0:
                    self._save_checkpoint()

                self.state.global_step += 1

        # Final checkpoint
        self._save_checkpoint(is_final=True)
        logger.info(f"Training complete. Total time: {self.state.elapsed_time:.1f}s")

    def _train_step(
        self,
        batch: dict[str, Any],
        weights: dict[str, float],
    ) -> dict[str, Tensor]:
        """Single training step.

        Args:
            batch: Data batch with images, poses, intrinsics, supervision.
            weights: Loss weights for current stage.

        Returns:
            Dict of loss components.
        """
        self.encoder.train()
        self.occ_head.train()
        self.sem_head.train()

        # Move data to device
        images = batch["images"].to(self.device)  # (B, V, 3, H, W)
        poses = batch["poses"].to(self.device)    # (B, V, 4, 4)
        intrinsics = batch["intrinsics"].to(self.device)  # (B, V, 3, 3)

        # ReSplat encoder forward pass
        # This produces Gaussian parameters
        context = {
            "image": images,
            "extrinsics": poses,
            "intrinsics": intrinsics,
        }
        encoder_output = self.encoder(context, self.state.global_step, deterministic=False)
        gaussians = encoder_output["gaussians"]

        # Extract Gaussian parameters
        means = gaussians.means        # (B, G, 3)
        scales = gaussians.scales      # (B, G, 3)
        rotations = gaussians.quats    # (B, G, 4)
        opacities = gaussians.opacities  # (B, G)

        B, G, _ = means.shape

        # Flatten for head predictions
        means_flat = means.reshape(-1, 3)       # (B*G, 3)
        scales_flat = scales.reshape(-1, 3)     # (B*G, 3)
        rotations_flat = rotations.reshape(-1, 4)  # (B*G, 4)

        # Predict occupancy and semantics
        pred_occ = self.occ_head(means_flat, scales_flat, rotations_flat)  # (B*G,)
        pred_sem = self.sem_head(means_flat, scales_flat, rotations_flat)  # (B*G, K)

        # Reshape back
        pred_occ = pred_occ.reshape(B, G)       # (B, G)
        pred_sem = pred_sem.reshape(B, G, -1)   # (B, G, K)

        # Compute losses per batch element
        total_losses = {}
        for b in range(B):
            # Get supervision for this sample
            if "labels" in batch:
                labels = batch["labels"]
                if isinstance(labels, dict):
                    occ_mask = labels["occupied"][b].squeeze().to(self.device)
                    free_mask = labels["free"][b].squeeze().to(self.device)
                    sem_labels = labels.get("semantic")
                    if sem_labels is not None:
                        sem_labels = sem_labels[b].squeeze().to(self.device)
                else:
                    # Pre-computed GaussianLabels
                    occ_mask = labels.occupied[b].to(self.device)
                    free_mask = labels.free[b].to(self.device)
                    sem_labels = labels.semantic_labels
                    if sem_labels is not None:
                        sem_labels = sem_labels[b].to(self.device)
            else:
                # Online labeling (slower, but no pre-computation needed)
                vggt_depth = batch["depth"][b, 0].to(self.device)  # (H, W)
                pose = poses[b, 0]  # (4, 4)
                K = intrinsics[b, 0]  # (3, 3)

                labels = label_gaussians_by_vggt_projection(
                    means_flat[b*G:(b+1)*G],
                    gaussians.covariances.reshape(-1, 3, 3)[b*G:(b+1)*G],
                    vggt_depth, pose, K,
                    kappa=self.config.kappa,
                )
                occ_mask = labels.occupied
                free_mask = labels.free
                sem_labels = labels.semantic_labels

            # Surface points for Chamfer loss
            surface_points = None
            if "surface_points" in batch:
                surface_points = batch["surface_points"][b].to(self.device)

            # Compute loss for this sample
            sample_losses = compute_total_loss(
                gaussian_means=means_flat[b*G:(b+1)*G],
                predicted_occupancy=pred_occ[b],
                semantic_logits=pred_sem[b] if pred_sem.dim() == 3 else None,
                vggt_points=surface_points,
                occupied_mask=occ_mask,
                free_mask=free_mask,
                semantic_labels=sem_labels,
                **weights,
                focal_gamma=self.config.focal_gamma,
                hinge_epsilon=self.config.hinge_epsilon,
                ssim_weight=self.config.ssim_weight,
                label_smoothing=self.config.label_smoothing,
            )

            # Accumulate
            for k, v in sample_losses.items():
                if k not in total_losses:
                    total_losses[k] = []
                total_losses[k].append(v)

        # Average over batch
        avg_losses = {k: torch.stack(v).mean() for k, v in total_losses.items()}

        # Backward pass
        self.optimizer.zero_grad()
        avg_losses["total"].backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) +
            list(self.occ_head.parameters()) +
            list(self.sem_head.parameters()),
            self.config.gradient_clip_val,
        )

        self.optimizer.step()

        # Track losses
        loss_dict = {k: v.item() for k, v in avg_losses.items()}
        self.state.losses_history.append(loss_dict)

        return avg_losses

    def _get_current_stage(self) -> int:
        """Determine current training stage based on progress."""
        progress = self.state.global_step / max(self.config.total_steps, 1)
        if progress < self.config.stage1_ratio:
            return 1
        elif progress < self.config.stage1_ratio + self.config.stage2_ratio:
            return 2
        else:
            return 3

    def _log_losses(
        self,
        losses: dict[str, Tensor],
        weights: dict[str, float],
    ) -> None:
        """Log current losses and stage info."""
        stage = self.state.stage
        step = self.state.global_step

        loss_str = " | ".join(
            f"{k}: {v.item():.4f}" for k, v in losses.items() if k != "total"
        )
        weight_str = " | ".join(f"{k}={v}" for k, v in weights.items())

        logger.info(
            f"[Step {step:6d}] Stage {stage} | "
            f"Total: {losses['total'].item():.4f} | "
            f"{loss_str} | "
            f"Weights: {weight_str}"
        )

    def _evaluate(self) -> None:
        """Run evaluation on validation set."""
        if self.val_loader is None:
            return

        self.encoder.eval()
        self.occ_head.eval()
        self.sem_head.eval()

        eval_losses = []
        with torch.no_grad():
            for batch in self.val_loader:
                # Simplified eval — compute losses without gradient
                weights = get_stage_weights(
                    self.state.global_step,
                    self.config.total_steps,
                    self.config.stage_weights,
                )
                losses = self._train_step(batch, weights)
                eval_losses.append({k: v.item() for k, v in losses.items()})

        avg_eval = {
            k: sum(d[k] for d in eval_losses) / len(eval_losses)
            for k in eval_losses[0]
        }

        logger.info(f"[Eval @ Step {self.state.global_step}] {avg_eval}")

        # Save best model
        if self.config.save_best and avg_eval["total"] < self.state.best_metric:
            self.state.best_metric = avg_eval["total"]
            self._save_checkpoint(is_best=True)

    def _save_checkpoint(
        self,
        is_best: bool = False,
        is_final: bool = False,
    ) -> None:
        """Save training checkpoint."""
        checkpoint = {
            "encoder": self.encoder.state_dict(),
            "occ_head": self.occ_head.state_dict(),
            "sem_head": self.sem_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "state": {
                "global_step": self.state.global_step,
                "epoch": self.state.epoch,
                "best_metric": self.state.best_metric,
                "stage": self.state.stage,
            },
            "config": self.config,
        }

        # Regular checkpoint
        path = self.output_dir / f"checkpoint_step_{self.state.global_step:06d}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")

        # Best checkpoint
        if is_best:
            best_path = self.output_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint: {best_path}")

        # Final checkpoint
        if is_final:
            final_path = self.output_dir / "checkpoint_final.pt"
            torch.save(checkpoint, final_path)
            logger.info(f"Saved final checkpoint: {final_path}")

    def _load_checkpoint(self, path: str) -> None:
        """Resume from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.encoder.load_state_dict(checkpoint["encoder"])
        self.occ_head.load_state_dict(checkpoint["occ_head"])
        self.sem_head.load_state_dict(checkpoint["sem_head"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])

        state = checkpoint["state"]
        self.state.global_step = state["global_step"]
        self.state.epoch = state["epoch"]
        self.state.best_metric = state["best_metric"]
        self.state.stage = state["stage"]

        logger.info(f"Resumed from {path} at step {self.state.global_step}")

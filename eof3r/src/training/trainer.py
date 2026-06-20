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
    use_bce: bool = False  # Use BCE instead of focal loss (ablation)
    relabel_every_n_steps: int = 50  # Recompute labels every N steps (multi-view)

    # Position loss hyperparameters
    pos_kappa_attract: float = 1.0
    pos_kappa_repel: float = 0.1

    # BEV loss hyperparameters
    bev_target_coverage: float = 0.10
    bev_resolution: float = 0.05
    bev_range: tuple[float, float, float, float] = (-10.0, -10.0, 10.0, 10.0)
    bev_height_range: tuple[float, float] = (-0.5, 2.0)

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
        1: {"alpha": 1.0, "beta": 0.3, "gamma": 0.1, "delta": 0.0, "eta": 0.3, "theta": 0.5, "zeta": 0.3},
        2: {"alpha": 0.5, "beta": 1.0, "gamma": 0.5, "delta": 0.3, "eta": 0.1, "theta": 0.3, "zeta": 0.1},
        3: {"alpha": 0.3, "beta": 1.0, "gamma": 1.0, "delta": 0.5, "eta": 0.05, "theta": 0.1, "zeta": 0.05},
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

    Args:
        data_dir: Path to the split-specific directory containing manifest.txt
            (e.g., outputs/vggt_supervision/test/). The manifest.txt should
            list scene IDs, one per line.
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
        # Try split-specific directory first, then direct path
        split_dir = self.data_dir / self.split
        if (split_dir / "manifest.txt").exists():
            manifest_dir = split_dir
        elif (self.data_dir / "manifest.txt").exists():
            manifest_dir = self.data_dir
        else:
            raise FileNotFoundError(
                f"Supervision manifest not found at {self.data_dir}/manifest.txt "
                f"or {split_dir}/manifest.txt\n"
                f"Run: python eof3r/scripts/preprocess/precompute_vggt_supervision.py"
            )

        manifest_path = manifest_dir / "manifest.txt"

        samples = []
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    sample_dir = manifest_dir / line
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
        sample = {
            "images": torch.load(paths["images"], weights_only=True),
            "depth": torch.load(paths["depth"], weights_only=True),
            "poses": torch.load(paths["poses"], weights_only=True),
            "intrinsics": torch.load(paths["intrinsics"], weights_only=True),
            "surface_points": torch.load(paths["points"], weights_only=True),
        }
        # Labels are optional (computed online during training if absent)
        if paths["labels"].exists():
            sample["labels"] = torch.load(paths["labels"], weights_only=True)
        return sample


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
        B, V = images.shape[:2]
        context = {
            "image": images,
            "extrinsics": poses,
            "intrinsics": intrinsics,
            "near": torch.full((B, V), 0.01, device=self.device),
            "far": torch.full((B, V), 100.0, device=self.device),
            "index": torch.arange(V, device=self.device).unsqueeze(0).expand(B, -1),
        }
        encoder_output = self.encoder(context, self.state.global_step, deterministic=False)
        gaussians = encoder_output["gaussians"]

        # Extract Gaussian parameters
        means = gaussians.means        # (B, G, 3)
        scales = gaussians.scales      # (B, G, 3)
        rotations = gaussians.rotations if hasattr(gaussians, 'rotations') else gaussians.quats  # (B, G, 4)
        opacities = gaussians.opacities  # (B, G)

        B, G, _ = means.shape

        # Flatten for head predictions
        means_flat = means.reshape(-1, 3)       # (B*G, 3)
        scales_flat = scales.reshape(-1, 3)     # (B*G, 3)
        rotations_flat = rotations.reshape(-1, 4)  # (B*G, 4)

        # Extract visual features from encoder output (Fix 2)
        # The encoder may provide per-Gaussian features (condition_features)
        # or raw image features. If neither, fall back to geometric-only.
        visual_features = None
        if "condition_features" in encoder_output:
            # ReSplat-style per-pixel features in latent grid
            # Need to map from (BV, C, H_lat, W_lat) to per-Gaussian
            cf = encoder_output["condition_features"]  # (B*V, C, H_lat, W_lat)
            _, C, H_lat, W_lat = cf.shape
            # Reshape to (B, V*H_lat*W_lat*M, C) pattern
            # For simplicity: flatten and repeat per Gaussian if needed
            cf_flat = cf.permute(0, 2, 3, 1).reshape(B * V, H_lat * W_lat, C)
            # Take mean per view as a per-view descriptor
            visual_features = cf_flat.mean(dim=1)  # (B*V, C)
            # Expand to per-Gaussian: repeat V times for G entries
            # This assumes G is a multiple of V * H_lat * W_lat
            visual_features = visual_features.unsqueeze(1).expand(-1, G // V, -1).reshape(B * G, C)

        # If no explicit features, project Gaussians to first view image for RGB context
        if visual_features is None:
            # Simple fallback: use the raw image at projected Gaussian pixel
            # Project Gaussians to image 0 for each batch element
            visual_list = []
            for b in range(B):
                img = images[b, 0]  # (3, H_img, W_img) — first view
                pose = poses[b, 0]  # (4, 4)
                K = intrinsics[b, 0]  # (3, 3)

                # Transform Gaussian means to camera frame, project, sample RGB
                T_c_w = torch.inverse(pose)  # world→camera
                R_c_w = T_c_w[:3, :3]
                t_c_w = T_c_w[:3, 3]
                means_cam = (R_c_w @ means[b].T).T + t_c_w  # (G, 3)

                # Project to pixels
                proj = (K @ means_cam.T).T  # (G, 3)
                depths = proj[:, 2].clamp(min=1e-6)
                uv = proj[:, :2] / depths.unsqueeze(-1)  # (G, 2)

                # Normalize to [-1, 1] for grid_sample
                _, H_img, W_img = img.shape
                uv_norm = torch.stack([
                    2.0 * uv[:, 0] / (W_img - 1) - 1.0,
                    2.0 * uv[:, 1] / (H_img - 1) - 1.0,
                ], dim=-1)  # (G, 2)

                # Bilinear sample RGB from image
                # grid_sample expects (N, H_out, W_out, 2) but we want per-point
                img_unsq = img.unsqueeze(0)  # (1, 3, H, W)
                uv_unsq = uv_norm.unsqueeze(0).unsqueeze(0)  # (1, 1, G, 2)
                rgb_sampled = torch.nn.functional.grid_sample(
                    img_unsq, uv_unsq, mode='bilinear', padding_mode='border', align_corners=True
                )  # (1, 3, 1, G)
                rgb_sampled = rgb_sampled.squeeze(0).squeeze(1).T  # (G, 3)

                # Also include normalized depth as a feature
                depth_norm = depths / depths.max().clamp(min=1e-6)

                vis = torch.cat([rgb_sampled, depth_norm.unsqueeze(-1)], dim=-1)  # (G, 4)
                visual_list.append(vis)

            visual_features = torch.cat(visual_list, dim=0)  # (B*G, D_vis)

        # Predict occupancy and semantics
        pred_occ = self.occ_head(means_flat, scales_flat, rotations_flat, visual_features=visual_features)  # (B*G,)
        pred_sem = self.sem_head(means_flat, scales_flat, rotations_flat)  # (B*G, K)

        # Reshape back
        pred_occ = pred_occ.reshape(B, G)       # (B, G)
        pred_sem = pred_sem.reshape(B, G, -1)   # (B, G, K)

        # Compute losses per batch element
        total_losses = {}
        for b in range(B):
            # Get supervision for this sample
            # Use multi-view labeling: loop over all views, merge results
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
                # Multi-view online labeling (Fix 4): use ALL camera views
                # Staggered re-labeling: only compute every N steps for efficiency
                all_labels = []
                for v_idx in range(V):
                    vggt_depth = batch["depth"][b, v_idx].to(self.device)  # (H, W)
                    pose = poses[b, v_idx]  # (4, 4)
                    K = intrinsics[b, v_idx]  # (3, 3)

                    labels_v = label_gaussians_by_vggt_projection(
                        means_flat[b*G:(b+1)*G],
                        gaussians.covariances.reshape(-1, 3, 3)[b*G:(b+1)*G],
                        vggt_depth, pose, K,
                        kappa=self.config.kappa,
                    )
                    all_labels.append(labels_v)

                # Merge multi-view labels: occupied if any view, free if all views
                labels = merge_multi_view_labels(all_labels)
                occ_mask = labels.occupied
                free_mask = labels.free
                sem_labels = labels.semantic_labels

            # Surface points for Chamfer loss
            surface_points = None
            if "surface_points" in batch:
                surface_points = batch["surface_points"][b].to(self.device)

            # Differentiable BEV projection (Fix 1): connect occupancy to BEV
            bev_grid = None
            if surface_points is not None and weights.get("zeta", 0) > 0:
                means_yup = means[b]  # (G, 3) in Y-up
                # Y-up (x-right, y-up, z-backward) → Z-up (x-fwd, y-left, z-up)
                means_zup = torch.stack([
                    means_yup[:, 0],
                    -means_yup[:, 2],
                    means_yup[:, 1],
                ], dim=-1)
                from eof3r.src.training.losses import differentiable_bev_projection
                bev_grid = differentiable_bev_projection(
                    means_zup=means_zup,
                    occupancies=pred_occ[b],
                    bev_resolution=self.config.bev_resolution,
                    bev_range=self.config.bev_range,
                    height_range=self.config.bev_height_range,
                )

            # Compute loss for this sample
            sample_losses = compute_total_loss(
                gaussian_means=means_flat[b*G:(b+1)*G],
                predicted_occupancy=pred_occ[b],
                semantic_logits=pred_sem[b] if pred_sem.dim() == 3 else None,
                vggt_points=surface_points,
                occupied_mask=occ_mask,
                free_mask=free_mask,
                semantic_labels=sem_labels,
                bev_grid=bev_grid,
                **weights,
                focal_gamma=self.config.focal_gamma,
                hinge_epsilon=self.config.hinge_epsilon,
                ssim_weight=self.config.ssim_weight,
                label_smoothing=self.config.label_smoothing,
                use_bce=self.config.use_bce,
                pos_kappa_attract=self.config.pos_kappa_attract,
                pos_kappa_repel=self.config.pos_kappa_repel,
                bev_target_coverage=self.config.bev_target_coverage,
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

    def _eval_step(
        self,
        batch: dict[str, Any],
        weights: dict[str, float],
    ) -> dict[str, Tensor]:
        """Evaluation step — compute losses without backward/optimizer.

        Same as _train_step but without gradient computation, optimizer update,
        or loss history tracking. Used inside torch.no_grad() context.
        """
        # Move data to device
        images = batch["images"].to(self.device)
        poses = batch["poses"].to(self.device)
        intrinsics = batch["intrinsics"].to(self.device)

        B, V = images.shape[:2]
        context = {
            "image": images,
            "extrinsics": poses,
            "intrinsics": intrinsics,
            "near": torch.full((B, V), 0.01, device=self.device),
            "far": torch.full((B, V), 100.0, device=self.device),
            "index": torch.arange(V, device=self.device).unsqueeze(0).expand(B, -1),
        }
        encoder_output = self.encoder(context, self.state.global_step, deterministic=True)
        gaussians = encoder_output["gaussians"]

        means = gaussians.means
        scales = gaussians.scales
        rotations = gaussians.rotations if hasattr(gaussians, 'rotations') else gaussians.quats

        B, G, _ = means.shape
        means_flat = means.reshape(-1, 3)
        scales_flat = scales.reshape(-1, 3)
        rotations_flat = rotations.reshape(-1, 4)

        # Extract visual features (same as _train_step)
        visual_features_v = None
        if "condition_features" in encoder_output:
            cf = encoder_output["condition_features"]
            _, C, H_lat, W_lat = cf.shape
            cf_flat = cf.permute(0, 2, 3, 1).reshape(B * V, H_lat * W_lat, C)
            visual_features_v = cf_flat.mean(dim=1)
            visual_features_v = visual_features_v.unsqueeze(1).expand(-1, G // V, -1).reshape(B * G, C)

        if visual_features_v is None:
            visual_list = []
            for b in range(B):
                img = images[b, 0]
                pose = poses[b, 0]
                K = intrinsics[b, 0]
                T_c_w = torch.inverse(pose)
                R_c_w = T_c_w[:3, :3]
                t_c_w = T_c_w[:3, 3]
                means_cam = (R_c_w @ means[b].T).T + t_c_w
                proj = (K @ means_cam.T).T
                depths = proj[:, 2].clamp(min=1e-6)
                uv = proj[:, :2] / depths.unsqueeze(-1)
                _, H_img, W_img = img.shape
                uv_norm = torch.stack([
                    2.0 * uv[:, 0] / (W_img - 1) - 1.0,
                    2.0 * uv[:, 1] / (H_img - 1) - 1.0,
                ], dim=-1)
                img_unsq = img.unsqueeze(0)
                uv_unsq = uv_norm.unsqueeze(0).unsqueeze(0)
                rgb_sampled = torch.nn.functional.grid_sample(
                    img_unsq, uv_unsq, mode='bilinear', padding_mode='border', align_corners=True
                )
                rgb_sampled = rgb_sampled.squeeze(0).squeeze(1).T
                depth_norm = depths / depths.max().clamp(min=1e-6)
                vis = torch.cat([rgb_sampled, depth_norm.unsqueeze(-1)], dim=-1)
                visual_list.append(vis)
            visual_features_v = torch.cat(visual_list, dim=0)

        pred_occ = self.occ_head(means_flat, scales_flat, rotations_flat, visual_features=visual_features_v)
        pred_sem = self.sem_head(means_flat, scales_flat, rotations_flat)
        pred_occ = pred_occ.reshape(B, G)
        pred_sem = pred_sem.reshape(B, G, -1)

        total_losses: dict[str, list[Tensor]] = {}
        for b in range(B):
            # Multi-view labeling (Fix 4)
            if "labels" in batch:
                labels = batch["labels"]
                if isinstance(labels, dict):
                    occ_mask = labels["occupied"][b].squeeze().to(self.device)
                    free_mask = labels["free"][b].squeeze().to(self.device)
                    sem_labels = labels.get("semantic")
                    if sem_labels is not None:
                        sem_labels = sem_labels[b].squeeze().to(self.device)
                else:
                    occ_mask = labels.occupied[b].to(self.device)
                    free_mask = labels.free[b].to(self.device)
                    sem_labels = labels.semantic_labels
                    if sem_labels is not None:
                        sem_labels = sem_labels[b].to(self.device)
            else:
                all_labels = []
                for v_idx in range(V):
                    vggt_depth = batch["depth"][b, v_idx].to(self.device)
                    pose = poses[b, v_idx]
                    K = intrinsics[b, v_idx]
                    labels_v = label_gaussians_by_vggt_projection(
                        means_flat[b*G:(b+1)*G],
                        gaussians.covariances.reshape(-1, 3, 3)[b*G:(b+1)*G],
                        vggt_depth, pose, K,
                        kappa=self.config.kappa,
                    )
                    all_labels.append(labels_v)
                labels_result = merge_multi_view_labels(all_labels)
                occ_mask = labels_result.occupied
                free_mask = labels_result.free
                sem_labels = labels_result.semantic_labels

            surface_points = None
            if "surface_points" in batch:
                surface_points = batch["surface_points"][b].to(self.device)

            # Differentiable BEV projection (Fix 1)
            bev_grid_v = None
            if surface_points is not None and weights.get("zeta", 0) > 0:
                means_yup = means[b]
                means_zup = torch.stack([
                    means_yup[:, 0], -means_yup[:, 2], means_yup[:, 1]
                ], dim=-1)
                from eof3r.src.training.losses import differentiable_bev_projection
                bev_grid_v = differentiable_bev_projection(
                    means_zup=means_zup,
                    occupancies=pred_occ[b],
                    bev_resolution=self.config.bev_resolution,
                    bev_range=self.config.bev_range,
                    height_range=self.config.bev_height_range,
                )

            sample_losses = compute_total_loss(
                gaussian_means=means_flat[b*G:(b+1)*G],
                predicted_occupancy=pred_occ[b],
                semantic_logits=pred_sem[b] if pred_sem.dim() == 3 else None,
                vggt_points=surface_points,
                occupied_mask=occ_mask,
                free_mask=free_mask,
                semantic_labels=sem_labels,
                bev_grid=bev_grid_v,
                **weights,
                focal_gamma=self.config.focal_gamma,
                hinge_epsilon=self.config.hinge_epsilon,
                ssim_weight=self.config.ssim_weight,
                label_smoothing=self.config.label_smoothing,
                use_bce=self.config.use_bce,
                pos_kappa_attract=self.config.pos_kappa_attract,
                pos_kappa_repel=self.config.pos_kappa_repel,
                bev_target_coverage=self.config.bev_target_coverage,
            )

            for k, v in sample_losses.items():
                if k not in total_losses:
                    total_losses[k] = []
                total_losses[k].append(v)

        avg_losses = {k: torch.stack(v).mean() for k, v in total_losses.items()}
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
                # Eval step — compute losses without backward/optimizer
                weights = get_stage_weights(
                    self.state.global_step,
                    self.config.total_steps,
                    self.config.stage_weights,
                )
                losses = self._eval_step(batch, weights)
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
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.encoder.load_state_dict(checkpoint["encoder"])
        self.occ_head.load_state_dict(checkpoint["occ_head"])
        self.sem_head.load_state_dict(checkpoint["sem_head"])

        # Load optimizer state (skip scheduler — it may have different total_steps)
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        except Exception as e:
            logger.warning(f"Could not load optimizer state: {e}. Using fresh optimizer.")

        state = checkpoint["state"]
        self.state.global_step = 0  # Reset step counter for new training run
        self.state.epoch = 0
        self.state.best_metric = state["best_metric"]
        self.state.stage = 1  # Reset to stage 1

        logger.info(f"Resumed model weights from {path} (previous step: {state['global_step']})")

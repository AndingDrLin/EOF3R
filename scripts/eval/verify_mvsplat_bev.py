#!/usr/bin/env python3
"""Verify that MVSplat feedforward 3DGS can output BEV-compatible Gaussians.

Run from the mvsplat directory:
  cd baselines/mvsplat && python ../../scripts/eval/verify_mvsplat_bev.py
"""

import os
import sys
from pathlib import Path

# Set up MVSplat root
MVSPLAT_ROOT = Path(__file__).resolve().parent.parent.parent / "baselines" / "mvsplat"
os.chdir(MVSPLAT_ROOT)
sys.path.insert(0, str(MVSPLAT_ROOT))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from src.model.types import Gaussians
from src.model.encoder import get_encoder
from src.model.decoder import get_decoder
from src.model.model_wrapper import ModelWrapper
from src.dataset.data_module import DataModule
from src.config import load_typed_root_config
from src.global_cfg import set_cfg
from src.misc.step_tracker import StepTracker
from src.model.encoder.common.gaussian_adapter import GaussianAdapter

import hydra
from omegaconf import OmegaConf


def load_model_simple():
    """Load MVSplat model using direct config construction."""
    # Use hydra compose with experiment override
    with hydra.initialize_config_dir(
        config_dir=str(MVSPLAT_ROOT / "config"),
        version_base=None,
    ):
        cfg_dict = hydra.compose(
            config_name="main",
            overrides=[
                "+experiment=re10k",
                "checkpointing.load=checkpoints/re10k.ckpt",
                "mode=test",
                "dataset/view_sampler=evaluation",
                "dataset.view_sampler.index_path=assets/evaluation_index_re10k.json",
                "dataset.skip_bad_shape=false",
                "test.compute_scores=false",
                "test.save_image=false",
                "test.save_video=false",
            ],
        )

    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    encoder, _ = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder, cfg.dataset)

    model_kwargs = {
        "optimizer_cfg": cfg.optimizer,
        "test_cfg": cfg.test,
        "train_cfg": cfg.train,
        "encoder": encoder,
        "encoder_visualizer": None,
        "decoder": decoder,
        "losses": [],
        "step_tracker": StepTracker(),
    }

    model = ModelWrapper.load_from_checkpoint(
        cfg.checkpointing.load, **model_kwargs, strict=True
    )
    model = model.cuda().eval()

    data_module = DataModule(cfg.dataset, cfg.data_loader, StepTracker(), global_rank=0)
    data_module.setup("test")
    test_loader = data_module.test_dataloader()
    batch = next(iter(test_loader))

    return model, batch, cfg


def extract_gaussians(model, batch):
    """Extract Gaussian parameters from a batch."""
    with torch.no_grad():
        # Move to GPU
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].cuda()
            elif isinstance(batch[k], dict):
                for kk in batch[k]:
                    if isinstance(batch[k][kk], torch.Tensor):
                        batch[k][kk] = batch[k][kk].cuda()
            elif isinstance(batch[k], list):
                batch[k] = [x.cuda() if isinstance(x, torch.Tensor) else x for x in batch[k]]

        gaussians = model.encoder(batch["context"], global_step=0, deterministic=False)

    return gaussians


def project_to_bev(means, opacities, scales=None,
                   height_range=(0.0, 2.5), grid_res=0.05, grid_half=5.0):
    """Project Gaussians to BEV occupancy grid."""
    means_np = means.cpu().numpy()
    opac_np = opacities.cpu().numpy()

    # Height filter (Y-up)
    mask = (means_np[:, 1] >= height_range[0]) & (means_np[:, 1] <= height_range[1])
    m = means_np[mask]
    o = opac_np[mask]

    if scales is not None:
        s = scales.cpu().numpy()[mask]
    else:
        s = np.full_like(m, 0.05)

    n = int(2 * grid_half / grid_res)
    bev = np.zeros((n, n), dtype=np.float32)

    for i in range(len(m)):
        x, y, z = m[i]
        sx, sy, sz = s[i]
        col = int((x + grid_half) / grid_res)
        row = int((z + grid_half) / grid_res)
        if 0 <= row < n and 0 <= col < n:
            r = max(sx, sz)
            rc = max(1, int(r / grid_res))
            for dr in range(-rc, rc + 1):
                for dc in range(-rc, rc + 1):
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < n and 0 <= nc < n:
                        dist = np.sqrt(dr**2 + dc**2) * grid_res
                        if dist <= r:
                            w = o[i] * np.exp(-0.5 * (dist / max(r, 1e-6))**2)
                            bev[nr, nc] = max(bev[nr, nc], w)

    return bev


def visualize(means, opacities, bev, save_dir):
    """Save visualizations."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    m = means.cpu().numpy()
    o = opacities.cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    # BEV occupancy
    im = axes[0, 0].imshow(bev, cmap='hot', origin='lower', vmin=0, vmax=1,
                            extent=[-5, 5, -5, 5])
    axes[0, 0].set_title('BEV Occupancy Grid')
    axes[0, 0].set_xlabel('X (m)'); axes[0, 0].set_ylabel('Z (m)')
    plt.colorbar(im, ax=axes[0, 0], shrink=0.8)

    # BEV binary
    axes[0, 1].imshow((bev > 0.3).astype(float), cmap='Greys', origin='lower',
                       extent=[-5, 5, -5, 5])
    axes[0, 1].set_title('BEV Footprint (threshold=0.3)')
    axes[0, 1].set_xlabel('X (m)'); axes[0, 1].set_ylabel('Z (m)')

    # XZ (top-down)
    axes[1, 0].scatter(m[:, 0], m[:, 2], s=0.3, alpha=0.2, c=o, cmap='viridis')
    axes[1, 0].set_title('Gaussians XZ (top-down)')
    axes[1, 0].set_xlabel('X (m)'); axes[1, 0].set_ylabel('Z (m)')
    axes[1, 0].set_aspect('equal'); axes[1, 0].grid(True, alpha=0.3)

    # XY (front)
    axes[1, 1].scatter(m[:, 0], m[:, 1], s=0.3, alpha=0.2, c=o, cmap='viridis')
    axes[1, 1].set_title('Gaussians XY (front view)')
    axes[1, 1].set_xlabel('X (m)'); axes[1, 1].set_ylabel('Y (m)')
    axes[1, 1].set_aspect('equal'); axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    p = save_dir / "mvsplat_bev_verify.png"
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    return p


def main():
    print("=" * 60)
    print("MVSplat BEV Verification")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # [1/4] Load model
    print("\n[1/4] Loading MVSplat model...")
    try:
        model, batch, cfg = load_model_simple()
        print(f"  Context views: {batch['context']['image'].shape[1]}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        return

    # [2/4] Extract Gaussians
    print("\n[2/4] Extracting Gaussians...")
    try:
        gaussians = extract_gaussians(model, batch)
        n = gaussians.means.shape[1]
        print(f"  Gaussians per scene: {n}")
        print(f"  Means: {gaussians.means.shape}")
        print(f"  Opacities: {gaussians.opacities.shape}")
        print(f"  Covariances: {gaussians.covariances.shape}")
        print(f"  Harmonics: {gaussians.harmonics.shape}")
        print(f"  Opacity range: [{gaussians.opacities.min():.4f}, {gaussians.opacities.max():.4f}]")
        print(f"  Opacity mean: {gaussians.opacities.mean():.4f}")

        # Check GaussianAdapter output (has scales)
        has_scales = hasattr(gaussians, 'scales') and gaussians.scales is not None
        print(f"  Has scales: {has_scales}")
        if has_scales:
            print(f"  Scales: {gaussians.scales.shape}")
            print(f"  Scale range: [{gaussians.scales.min():.4f}, {gaussians.scales.max():.4f}]")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        return

    # [3/4] BEV projection
    print("\n[3/4] Projecting to BEV...")
    try:
        bev = project_to_bev(
            gaussians.means[0], gaussians.opacities[0],
            scales=gaussians.scales[0] if has_scales else None,
        )
        occ_count = (bev > 0.3).sum()
        print(f"  BEV shape: {bev.shape}")
        print(f"  Occupied cells (>0.3): {occ_count}")
        print(f"  Coverage: {occ_count / bev.size * 100:.2f}%")
        print(f"  Max value: {bev.max():.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        return

    # [4/4] Visualize
    print("\n[4/4] Saving visualization...")
    save_dir = Path(__file__).resolve().parent.parent.parent / "outputs" / "eval"
    try:
        p = visualize(gaussians.means[0], gaussians.opacities[0], bev, save_dir)
        print(f"  Saved: {p}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        return

    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"✓ MVSplat loaded (re10k checkpoint)")
    print(f"✓ {n} Gaussians extracted per scene")
    print(f"✓ Parameters: means, covariances, opacities, harmonics")
    if has_scales:
        print(f"✓ Scales available for footprint estimation")
    print(f"✓ BEV grid: {bev.shape}, {occ_count} occupied cells")
    print(f"✓ Visualization: {p}")
    print()
    print("CONCLUSION: MVSplat CAN produce BEV-compatible Gaussians.")
    print("  → means (3D positions) → BEV footprint projection")
    print("  → opacities → occupancy thresholding")
    print("  → covariances/scales → footprint size estimation")


if __name__ == "__main__":
    main()

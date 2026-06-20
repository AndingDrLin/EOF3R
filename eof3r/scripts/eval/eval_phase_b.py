"""Phase B evaluation metrics for cross-model geometric distillation.

Computes:
1. Gaussian-surface distance (Chamfer) — are Gaussians near VGGT surfaces?
2. Occupancy accuracy — does occupancy head predict correctly?
3. BEV coverage — what fraction of BEV grid is occupied?
4. Label distribution — how many Gaussians are occupied/free/unknown?

Usage:
    python eof3r/scripts/eval/eval_phase_b.py \
        --checkpoint outputs/autolab/results/real_resplat_1k/checkpoint_final.pt \
        --supervision-dir outputs/autolab/vggt_all \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from eof3r.src.training.supervision import label_gaussians_by_vggt_projection
from eof3r.src.training.losses import chamfer_depth_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: str,
    supervision_dir: str,
    device: str = "cuda",
) -> dict:
    """Evaluate a trained Phase B model.

    Args:
        checkpoint_path: Path to checkpoint .pt file.
        supervision_dir: Path to VGGT supervision directory.
        device: Device for inference.

    Returns:
        Dict of evaluation metrics.
    """
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config")

    # Load encoder
    from eof3r.src.foreground.resplat_encoder import load_resplat_encoder
    encoder = load_resplat_encoder(
        str(PROJECT_ROOT / "baselines" / "resplat" / "pretrained"
            / "resplat-base-re10k-256x256-view2-b90d1b53.pth"),
        device=device,
    )

    # Load occupancy head from checkpoint
    from eof3r.src.training.heads import OccupancyHead
    occ_head = OccupancyHead(
        input_dim=10,
        hidden_dims=config.occ_hidden_dims if config else [64, 32],
    )
    occ_head.load_state_dict(ckpt["occ_head"])
    occ_head = occ_head.to(device).eval()

    # Load dataset
    from eof3r.src.training.trainer import VGGTSupervisionDataset
    dataset = VGGTSupervisionDataset(supervision_dir, split="test")
    logger.info(f"Evaluating on {len(dataset)} scenes")

    all_metrics = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        images = sample["images"].unsqueeze(0).to(device)
        poses = sample["poses"].unsqueeze(0).to(device)
        intrinsics = sample["intrinsics"].unsqueeze(0).to(device)
        surface_points = sample["surface_points"].to(device)

        B, V = images.shape[:2]
        context = {
            "image": images,
            "extrinsics": poses,
            "intrinsics": intrinsics,
            "near": torch.full((B, V), 0.01, device=device),
            "far": torch.full((B, V), 100.0, device=device),
            "index": torch.arange(V, device=device).unsqueeze(0).expand(B, -1),
        }

        # Forward pass
        output = encoder(context, 0, deterministic=True)
        gaussians = output["gaussians"]

        means = gaussians.means[0]  # (G, 3)
        scales = gaussians.scales[0]  # (G, 3)
        rotations = gaussians.rotations[0] if hasattr(gaussians, 'rotations') else gaussians.quats[0]
        opacities = gaussians.opacities[0]  # (G,)

        # Occupancy head predictions
        pred_occ = occ_head(means, scales, rotations)  # (G,)

        # Label Gaussians using VGGT supervision (first view)
        depth = sample["depth"][0].to(device)  # (H, W)
        pose = poses[0, 0]  # (4, 4)
        K = intrinsics[0, 0]  # (3, 3)

        # Build covariance from scales and rotations
        G = means.shape[0]
        cov = torch.zeros(G, 3, 3, device=device)
        cov[:, 0, 0] = scales[:, 0] ** 2
        cov[:, 1, 1] = scales[:, 1] ** 2
        cov[:, 2, 2] = scales[:, 2] ** 2

        labels = label_gaussians_by_vggt_projection(
            means, cov, depth, pose, K, kappa=3.0,
        )

        # 1. Gaussian-surface distance (subsampled Chamfer)
        n_points = min(surface_points.shape[0], 2048)
        n_gauss = min(G, 4096)
        idx_p = torch.randperm(surface_points.shape[0], device=device)[:n_points]
        idx_g = torch.randperm(G, device=device)[:n_gauss]

        diff = surface_points[idx_p].unsqueeze(1) - means[idx_g].unsqueeze(0)
        dist = (diff ** 2).sum(dim=-1)
        chamfer_fwd = dist.min(dim=1).values.mean().item()
        chamfer_bwd = dist.min(dim=0).values.mean().item()
        chamfer = chamfer_fwd + chamfer_bwd

        # 2. Occupancy accuracy
        labeled_mask = labels.occupied | labels.free
        if labeled_mask.any():
            gt_labels = labels.occupied.float()
            pred_labels = (pred_occ > 0.5).float()
            occ_accuracy = (pred_labels[labeled_mask] == gt_labels[labeled_mask]).float().mean().item()
            occ_precision = ((pred_labels == 1) & (gt_labels == 1) & labeled_mask).sum().item() / max((pred_labels == 1).sum().item(), 1)
            occ_recall = ((pred_labels == 1) & (gt_labels == 1) & labeled_mask).sum().item() / max(labels.occupied.sum().item(), 1)
        else:
            occ_accuracy = 0.0
            occ_precision = 0.0
            occ_recall = 0.0

        # 3. BEV coverage (simple XY projection)
        bev_resolution = 100
        xy = means[:, [0, 2]]  # (G, 2) — X and Z in Y-up
        occ_mask = pred_occ > 0.3
        if occ_mask.any():
            xy_occ = xy[occ_mask]
            # Normalize to [0, 1]
            xy_min = xy_occ.min(dim=0).values
            xy_max = xy_occ.max(dim=0).values
            xy_range = (xy_max - xy_min).clamp(min=1e-6)
            xy_norm = (xy_occ - xy_min) / xy_range
            # Bin to grid
            grid_x = (xy_norm[:, 0] * (bev_resolution - 1)).long().clamp(0, bev_resolution - 1)
            grid_y = (xy_norm[:, 1] * (bev_resolution - 1)).long().clamp(0, bev_resolution - 1)
            bev_grid = torch.zeros(bev_resolution, bev_resolution, device=device)
            bev_grid[grid_x, grid_y] = 1.0
            bev_coverage = bev_grid.sum().item() / (bev_resolution * bev_resolution)
        else:
            bev_coverage = 0.0

        # 4. Label distribution
        n_total = G
        n_occupied = labels.occupied.sum().item()
        n_free = labels.free.sum().item()
        n_unknown = labels.unknown.sum().item()

        metrics = {
            "scene_idx": idx,
            "num_gaussians": G,
            "chamfer_distance": chamfer,
            "chamfer_forward": chamfer_fwd,
            "chamfer_backward": chamfer_bwd,
            "occupancy_accuracy": occ_accuracy,
            "occupancy_precision": occ_precision,
            "occupancy_recall": occ_recall,
            "bev_coverage": bev_coverage,
            "label_occupied_pct": 100 * n_occupied / n_total,
            "label_free_pct": 100 * n_free / n_total,
            "label_unknown_pct": 100 * n_unknown / n_total,
            "opacity_mean": opacities.mean().item(),
            "pred_occ_mean": pred_occ.mean().item(),
        }
        all_metrics.append(metrics)

        logger.info(
            f"Scene {idx}: chamfer={chamfer:.4f}, occ_acc={occ_accuracy:.3f}, "
            f"bev_cov={bev_coverage:.4f}, "
            f"labels: occ={100*n_occupied/n_total:.1f}% free={100*n_free/n_total:.1f}%"
        )

    # Aggregate
    agg = {}
    for key in all_metrics[0]:
        if isinstance(all_metrics[0][key], (int, float)):
            vals = [m[key] for m in all_metrics]
            agg[f"{key}_mean"] = sum(vals) / len(vals)
            agg[f"{key}_std"] = (sum((v - agg[f"{key}_mean"])**2 for v in vals) / len(vals)) ** 0.5

    return {"per_scene": all_metrics, "aggregate": agg}


def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase B model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--supervision-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    metrics = evaluate_checkpoint(args.checkpoint, args.supervision_dir, args.device)

    output_path = args.output or str(Path(args.checkpoint).parent / "eval_metrics.json")
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {output_path}")

    # Print summary
    agg = metrics["aggregate"]
    print("\n" + "=" * 60)
    print("Phase B Evaluation Summary")
    print("=" * 60)
    print(f"Chamfer distance:      {agg['chamfer_distance_mean']:.4f} ± {agg['chamfer_distance_std']:.4f}")
    print(f"Occupancy accuracy:    {agg['occupancy_accuracy_mean']:.3f} ± {agg['occupancy_accuracy_std']:.3f}")
    print(f"BEV coverage:          {agg['bev_coverage_mean']:.4f} ± {agg['bev_coverage_std']:.4f}")
    print(f"Label occupied:        {agg['label_occupied_pct_mean']:.1f}% ± {agg['label_occupied_pct_std']:.1f}%")
    print(f"Label free:            {agg['label_free_pct_mean']:.1f}% ± {agg['label_free_pct_std']:.1f}%")
    print(f"Pred occupancy mean:   {agg['pred_occ_mean_mean']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

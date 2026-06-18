#!/usr/bin/env python3
"""Focused ablation study for EOF3R — mechanistic analysis.

Runs 4 variants across frame pairs to isolate each component's contribution:
  A_full:  YOLO+SAM2 + VGGT(scale) + MVSplat(aligned) — full pipeline
  B_noscale: VGGT without scale recovery — isolates scale contribution
  C_noalign: MVSplat with synthetic poses — isolates coord alignment contribution
  D_auto:   SAM2 automatic mode — isolates YOLO frontend contribution

Output: outputs/eval/ablation/ablation_summary.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_CODE_ROOT = _REPO_ROOT / "eof3r"
sys.path.insert(0, str(_CODE_ROOT))

import matplotlib
matplotlib.use("Agg")
import torch

# Paths
MVSPLAT_ROOT = _REPO_ROOT / "baselines" / "mvsplat"
CKPT = str(MVSPLAT_ROOT / "checkpoints" / "re10k.ckpt")
PUBLIC = _REPO_ROOT / "data" / "public" / "re10k_samples"
OUTDIR = _REPO_ROOT / "outputs" / "eval" / "ablation"


def log(msg: str) -> None:
    print(f"  {msg}")


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    import yaml
    with open(_CODE_ROOT / "configs" / "default.yaml") as f:
        config = yaml.safe_load(f)

    # ---- Detect model availability ----
    from src.segmentation import SAM2, SAM2Stub
    from src.background import VGGT, VGGTStub

    REAL_SAM2 = SAM2 is not SAM2Stub
    REAL_VGGT = VGGT is not VGGTStub
    HAS_GPU = torch.cuda.is_available()

    log(f"SAM2: {'REAL' if REAL_SAM2 else 'STUB'}, VGGT: {'REAL' if REAL_VGGT else 'STUB'}, GPU: {'YES' if HAS_GPU else 'NO'}")
    if not os.path.exists(CKPT):
        log("ERROR: MVSplat checkpoint missing. Aborting.")
        return

    # ---- Load images ----
    from PIL import Image
    images_uint8 = []
    for i in range(4):
        fp = PUBLIC / f"frame_{i:02d}.png"
        if fp.exists():
            images_uint8.append(np.array(Image.open(fp).convert("RGB")))
    log(f"Loaded {len(images_uint8)} frames ({images_uint8[0].shape})")

    # ---- Build all models ONCE ----
    log("Building models...")
    t0 = time.perf_counter()

    # SAM2 ×2
    sam2_base = {"model_size": "base", "min_mask_area": 500}
    seg_yolo = SAM2(**sam2_base, use_yolo=True)
    seg_auto = SAM2(**sam2_base, use_yolo=False)
    if REAL_SAM2:
        seg_yolo.build()
        seg_auto.build()

    # VGGT ×2
    vggt_base = {"max_resolution": 512, "estimate_ground": True, "estimate_drivable": True}
    bg_scaled = VGGT(**vggt_base, known_camera_height_m=1.5)
    bg_noscale = VGGT(**vggt_base, known_camera_height_m=None)
    if REAL_VGGT:
        bg_scaled.build()
        bg_noscale.build()

    # MVSplat ×1
    from src.foreground import MVSplatWrapper
    fg = MVSplatWrapper()
    fg.build(checkpoint_path=CKPT, mvsplat_root=str(MVSPLAT_ROOT))

    log(f"Models built in {time.perf_counter() - t0:.1f}s")

    # ---- Run ablation variants ----
    VARIANTS = [
        ("A_full",       seg_yolo, bg_scaled,  True),
        ("B_noscale",    seg_yolo, bg_noscale, True),
        ("C_noalign",    seg_yolo, bg_scaled,  False),
        ("D_sam2_auto",  seg_auto, bg_scaled,  True),
    ]

    # Frame pairings: 3 diverse pairs from 4 available frames
    FRAME_PAIRS = [
        ("pair_01", [0, 1]),
        ("pair_02", [0, 2]),
        ("pair_23", [1, 3]),
    ]

    all_results: list[dict] = []

    for pair_name, indices in FRAME_PAIRS:
        imgs = [images_uint8[i] for i in indices]
        log(f"\n--- {pair_name} (frames {indices}) ---")

        for var_name, seg, bg, use_alignment in VARIANTS:
            t_run = time.perf_counter()
            try:
                metrics = run_one(
                    images=imgs, seg=seg, bg=bg, fg=fg,
                    use_alignment=use_alignment, var_name=var_name,
                    config=config,
                )
                metrics["pair"] = pair_name
                metrics["variant"] = var_name
                metrics["wall_time"] = round(time.perf_counter() - t_run, 1)
                all_results.append(metrics)

                log(f"{var_name:<16} | obj={metrics['num_objects']:2d} {metrics['labels'][:3]} | "
                    f"BEVcov={metrics['bev_coverage_t0.3']:.4f} | "
                    f"iou={metrics['fg_bg_iou']:.4f} | "
                    f"conflict={metrics['drivable_conflict']:.4f} | "
                    f"free={metrics['costmap_free_pct']:.3f} lethal={metrics['costmap_lethal_pct']:.3f} | "
                    f"scale={metrics['scale_factor']:.1f} | "
                    f"sem={'Y' if metrics['has_semantics'] else 'N'} | "
                    f"t={metrics['total_time_s']:.1f}s")
            except Exception as e:
                log(f"{var_name:<16} | FAILED: {e}")
                all_results.append({"variant": var_name, "pair": pair_name, "status": "FAILED", "error": str(e)})
                import traceback
                traceback.print_exc()

            if HAS_GPU:
                torch.cuda.empty_cache()

    # ---- Aggregate ----
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS (mean over 3 frame pairs)")
    print("=" * 70)

    from collections import defaultdict
    agg: dict[str, list] = defaultdict(list)
    for r in all_results:
        if "bev_coverage_t0.3" in r:
            agg[r["variant"]].append(r)

    # Also collect per-variant labels for qualitative analysis
    label_samples: dict[str, list] = defaultdict(list)

    print(f"\n{'Variant':<18} {'BEVcov':>8} {'FG/BGIoU':>9} {'Conflict':>9} {'Free%':>7} {'Lethal%':>8} {'Obj':>5} {'Sem':>4} {'Scale':>6} {'Time':>6}")
    print("-" * 90)
    for var_name, results in sorted(agg.items()):
        avg = lambda k: np.mean([r[k] for r in results])
        sem = "Y" if any(r["has_semantics"] for r in results) else "N"
        for r in results:
            label_samples[var_name].extend(r["labels"])
        print(f"{var_name:<18} {avg('bev_coverage_t0.3'):>8.4f} {avg('fg_bg_iou'):>9.4f} {avg('drivable_conflict'):>9.4f} {avg('costmap_free_pct'):>7.3f} {avg('costmap_lethal_pct'):>8.3f} {avg('num_objects'):>5.1f} {sem:>4} {avg('scale_factor'):>6.1f} {avg('total_time_s'):>5.1f}s")

    # ---- Summary JSON ----
    summary = {
        "variants": list(agg.keys()),
        "num_pairs": len(FRAME_PAIRS),
        "per_variant": {
            v: {
                "avg_bev_cov": round(float(np.mean([r["bev_coverage_t0.3"] for r in results])), 6),
                "avg_fg_bg_iou": round(float(np.mean([r["fg_bg_iou"] for r in results])), 6),
                "avg_drivable_conflict": round(float(np.mean([r["drivable_conflict"] for r in results])), 6),
                "avg_costmap_free": round(float(np.mean([r["costmap_free_pct"] for r in results])), 4),
                "avg_costmap_lethal": round(float(np.mean([r["costmap_lethal_pct"] for r in results])), 4),
                "avg_num_objects": round(float(np.mean([r["num_objects"] for r in results])), 1),
                "avg_total_time_s": round(float(np.mean([r["total_time_s"] for r in results])), 1),
                "has_semantics": any(r["has_semantics"] for r in results),
                "typical_labels": list(set(label_samples.get(v, [])))[:10],
            }
            for v, results in agg.items() if results
        },
    }
    with open(OUTDIR / "ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {OUTDIR / 'ablation_summary.json'}")
    print("Done.")


# ===========================================================================
# Single pipeline run
# ===========================================================================
def run_one(
    images: list[np.ndarray],
    seg: object,
    bg: object,
    fg: object,
    use_alignment: bool,
    var_name: str,
    config: dict,
) -> dict:
    """Run full 5-stage pipeline. Returns metrics dict."""
    times: dict[str, float] = {}
    gpu_mb = 0.0

    # Stage 1 — Segmentation
    t0 = time.perf_counter()
    if var_name == "D_sam2_auto":
        seg_result = seg.segment(images[0])
    else:
        seg_result = seg.detect_and_segment(images[0], yolo_conf=0.35)
    times["seg"] = round(time.perf_counter() - t0, 4)

    # Stage 2 — Background
    t0 = time.perf_counter()
    bg_result = bg.infer(images[:min(len(images), 4)])
    scale_factor = bg_result.get("scale_factor", 1.0)
    times["bg"] = round(time.perf_counter() - t0, 4)

    # Stage 3 — Foreground
    t0 = time.perf_counter()
    H, W = 256, 256
    v = min(len(images), 2)
    from PIL import Image as PILImage
    resized = [np.array(PILImage.fromarray(im).resize((W, H), PILImage.LANCZOS), dtype=np.float32) / 255.0 for im in images[:v]]
    mv = torch.from_numpy(np.stack(resized)).permute(0, 3, 1, 2).unsqueeze(0).float().cuda()

    if use_alignment:
        wfc_ocv = bg_result.get("camera_poses_wfc_ocv")
        if wfc_ocv is not None and len(wfc_ocv) >= v:
            poses = torch.from_numpy(wfc_ocv[:v]).float().unsqueeze(0).cuda()
        else:
            poses = _default_poses(v, "cuda")
    else:
        poses = _default_poses(v, "cuda")

    K = torch.tensor([[[300, 0, W/2], [0, 300, H/2], [0, 0, 1]]], device="cuda").repeat(1, v, 1, 1).float()
    fg_result = fg.infer(mv, poses, K)
    g = fg_result["gaussians"]

    if use_alignment:
        from src.background.vggt_wrapper import _opencv_rdf_to_yup_points
        g = type("GaussianData", (), {
            "means": _opencv_rdf_to_yup_points(g.means),
            "opacities": g.opacities, "scales": g.scales,
            "covariances": g.covariances, "harmonics": g.harmonics,
            "rotations": g.rotations,
        })()
    times["fg"] = round(time.perf_counter() - t0, 4)
    if torch.cuda.is_available():
        gpu_mb = torch.cuda.max_memory_allocated() / 1024**2

    # Stage 4 — Fusion
    t0 = time.perf_counter()
    from src.fusion import BEVProjector
    proj = BEVProjector({"bev_range": "auto", "bev_target_cells": 400, "height_filter": [-1.0, 8.0]})
    bg_pm = bg_result["pointmap"][0]
    proj.set_bounds_from_points(fg_means_yup=g.means, bg_means_yup=bg_pm.reshape(-1, 3))

    fg_bev = proj.project_gaussians_to_bev(means=g.means, opacities=g.opacities, scales=g.scales)
    bg_pts = bg_pm.reshape(-1, 3)
    bg_bev = proj.project_gaussians_to_bev(
        means=bg_pts,
        opacities=np.random.rand(len(bg_pts)).astype(np.float32) * 0.5 + 0.3,
        scales=np.full((len(bg_pts), 3), 0.1, dtype=np.float32),
    )
    hb, wb = fg_bev.shape
    bg_drivable = BEVProjector._resize_to_grid(bg_result["drivable_mask"].astype(np.float32), hb, wb) > 0.5
    fused = proj.fuse_foreground_background(fg_bev, bg_bev, bg_drivable)
    times["fusion"] = round(time.perf_counter() - t0, 4)

    # Stage 5 — Costmap + Semantic
    t0 = time.perf_counter()
    from src.costmap import CostmapGenerator
    cg = CostmapGenerator(config.get("costmap", {}))
    sem_bev = None
    labels = seg_result.get("labels", [])
    if labels and labels[0] != "unknown":
        lbl_set = sorted(set(labels))
        l2id = {l: i for i, l in enumerate(lbl_set)}
        sids = np.array([l2id[l] for l in labels], dtype=np.int32)
        boxes = seg_result.get("boxes", np.zeros((0, 4)))
        if len(boxes) > 0:
            pm_h, pm_w = bg_pm.shape[:2]
            ih, iw = images[0].shape[:2]
            cents = []
            for box in boxes:
                cx = min(max(int((box[0]+box[2])/2*pm_w/iw), 0), pm_w-1)
                cy = min(max(int((box[1]+box[3])/2*pm_h/ih), 0), pm_h-1)
                cents.append(bg_pm[cy, cx])
            if cents:
                try:
                    sem_bev = proj.project_semantic_to_bev(
                        means=np.stack(cents), semantic_labels=sids,
                        opacities=seg_result.get("scores"),
                    )
                except Exception:
                    pass
    costmap, _ = cg.generate_costmap(fused, bev_semantic=sem_bev)
    times["costmap"] = round(time.perf_counter() - t0, 4)

    # Metrics
    m: dict = {}
    for t in [0.1, 0.3, 0.5, 0.7]:
        m[f"bev_coverage_t{t}"] = round(float((fused > t).sum()) / max(fused.size, 1), 6)
    occ = fused > 0.01
    m["bev_density"] = round(float(fused[occ].mean()) if occ.sum() > 0 else 0.0, 4)
    m["bev_extent_m2"] = round(float(occ.sum()) * 0.05 * 0.05, 2)
    m["num_gaussians"] = int(len(g.opacities))
    op = g.opacities
    m["opacity_mean"] = round(float(op.mean()), 4) if len(op) > 0 else 0
    for ai, al in enumerate(["x", "y", "z"]):
        m[f"gaussian_{al}_range"] = round(float(g.means[:, ai].max() - g.means[:, ai].min()), 2) if len(g.means) > 0 else 0

    fg_o = fg_bev > 0.3
    bg_o = bg_bev > 0.3
    drv = bg_drivable > 0.5
    m["fg_coverage"] = round(float(fg_o.sum()) / max(fg_bev.size, 1), 6)
    m["bg_coverage"] = round(float(bg_o.sum()) / max(bg_bev.size, 1), 6)
    inter = float((fg_o & bg_o).sum())
    union = float((fg_o | bg_o).sum())
    m["fg_bg_iou"] = round(inter / max(union, 1), 6)
    m["drivable_conflict"] = round(float((fg_o & drv).sum()) / max(fg_o.sum(), 1), 6)

    m["costmap_min"] = int(costmap.min())
    m["costmap_max"] = int(costmap.max())
    m["costmap_free_pct"] = round(float((costmap == 0).sum()) / costmap.size, 4)
    m["costmap_lethal_pct"] = round(float((costmap >= 254).sum()) / costmap.size, 4)
    m["costmap_completeness"] = round(float((costmap != 255).sum()) / costmap.size, 4)
    m["num_objects"] = len(labels)
    m["num_classes"] = len(set(labels))
    m["has_semantics"] = bool(labels and labels[0] != "unknown")
    m["labels"] = labels
    m["total_time_s"] = round(sum(times.values()), 3)
    m["scale_factor"] = round(scale_factor, 2)
    m["stage_times"] = times
    m["gpu_memory_mb"] = round(gpu_mb, 1)

    return m


def _default_poses(v: int, dev: str = "cpu") -> torch.Tensor:
    ps = [torch.eye(4, device=dev) for _ in range(v)]
    for vi in range(v):
        ps[vi][0, 3] = 0.3 * vi
    return torch.stack(ps).unsqueeze(0)


if __name__ == "__main__":
    main()

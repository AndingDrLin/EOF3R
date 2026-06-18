"""MVSplat feedforward Gaussian occupancy wrapper.

To use:
  1. git clone https://github.com/donydchen/mvsplat.git
  2. Set MVSPLAT_ROOT env var or pass mvsplat_root= to build().
  3. If MVSplat is not available, use --skip-mvsplat for synthetic Gaussians.

Provides:
  build()  — load MVSplat model from checkpoint
  infer()  — extract Gaussians from input images + poses + intrinsics
  extract_occupancy() — compute occupancy_alpha and BEV footprint
"""

from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


def _resolve_mvsplat_root(mvsplat_root: str | None = None) -> Path:
    """Resolve MVSplat root directory.

    Priority: 1) explicit parameter, 2) MVSPLAT_ROOT env var,
    3) local baselines/mvsplat/ for dev, 4) error.
    """
    if mvsplat_root is not None:
        p = Path(mvsplat_root)
        if p.exists():
            return p
        raise FileNotFoundError(f"MVSplat root not found: {mvsplat_root}")

    env_root = os.environ.get("MVSPLAT_ROOT")
    if env_root is not None:
        p = Path(env_root)
        if p.exists():
            return p
        raise FileNotFoundError(f"MVSPLAT_ROOT is set but path does not exist: {env_root}")

    # Development fallback: local baselines/ directory.
    dev_root = Path(__file__).resolve().parent.parent.parent.parent / "baselines" / "mvsplat"
    if dev_root.exists():
        return dev_root

    raise FileNotFoundError(
        "MVSplat not found. Either:\n"
        "  1. export MVSPLAT_ROOT=/path/to/mvsplat\n"
        "  2. Pass mvsplat_root='/path/to/mvsplat' to build()\n"
        "  3. Clone into baselines/mvsplat/ for development\n"
        "  Source: https://github.com/donydchen/mvsplat"
    )


def _ensure_mvsplat_on_path(mvsplat_root: str) -> tuple[str, list[str], dict]:
    """Register MVSplat source onto sys.path and chdir to MVSplat root.

    MVSplat's src/ package conflicts with our project's src/.
    We save the current sys.path + sys.modules state, strip everything
    except system paths, prepend MVSplat paths, and chdir to the MVSplat root.

    Returns:
        (previous_cwd, saved_path, saved_modules) — caller MUST restore these
        after the MVSplat operations complete.  See build() for the restore
        pattern.
    """
    import importlib

    root = Path(mvsplat_root)
    mvsplat_root_str = str(root)
    mvsplat_src = str(root / "src")

    # Save current state.
    saved_path = list(sys.path)
    saved_modules = {}
    for key in list(sys.modules.keys()):
        if key == "src" or key.startswith("src.") or key.startswith("config") or key.startswith("dataset") or key.startswith("model") or key.startswith("loss") or key.startswith("misc") or key.startswith("evaluation") or key.startswith("global_cfg"):
            saved_modules[key] = sys.modules.pop(key)
    previous_cwd = os.getcwd()

    # Clear all user paths, keep only system + MVSplat paths.
    system_paths = [p for p in sys.path if "site-packages" in p or "python3" in p or p in ("", "/usr/lib")]
    sys.path[:] = [mvsplat_root_str, mvsplat_src] + system_paths

    importlib.invalidate_caches()
    os.chdir(root)
    return previous_cwd, saved_path, saved_modules


# ---------------------------------------------------------------------------
# Lightweight dataclass mirroring MVSplat's GaussianAdapter.Gaussians
# but free of MVSplat import dependencies until runtime.
# ---------------------------------------------------------------------------


@dataclass
class GaussianData:
    """Normalised container for foreground Gaussian parameters.

    All tensors are on CPU (float32) after infer() returns.
    """

    means: np.ndarray  # (N, 3)  — xyz positions (Y-up, meters)
    opacities: np.ndarray  # (N,)   — opacity values [0, 1]
    scales: np.ndarray  # (N, 3)  — scale per axis
    covariances: np.ndarray  # (N, 3, 3) — world-space covariances
    harmonics: np.ndarray  # (N, 3, D_sh) — SH coefficients (may be discarded)
    rotations: np.ndarray | None  # (N, 4) — quaternions (w,x,y,z), None if unavailable

    # Per-Gaussian identity encoding (Gaussian Grouping style).
    # (N, D_id) float32 — learnable embedding, supervised by 2D masks.
    # D_id = 16 by default.  Initialized randomly, trained via differentiable
    # rasterization + classification head + cross-entropy against SAM masks.
    identity_encoding: np.ndarray | None = None  # None until add_identity_encoding() called


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


class MVSplatWrapper:
    """Feedforward Gaussian occupancy generator wrapping MVSplat.

    Usage::

        wrapper = MVSplatWrapper()
        wrapper.build(checkpoint_path="checkpoints/re10k.ckpt")
        result = wrapper.infer(images, poses, K)  # all torch.Tensors
        occ = wrapper.extract_occupancy(result["gaussians"], config={})
    """

    def __init__(self) -> None:
        self._model: torch.nn.Module | None = None
        self._gaussian_adapter: torch.nn.Module | None = None
        self._cfg: dict = {}
        self._loaded_checkpoint: str | None = None

    # ---- build ------------------------------------------------------------

    def build(
        self,
        checkpoint_path: str,
        config_overrides: dict | None = None,
        mvsplat_root: str | None = None,
    ) -> None:
        """Load MVSplat model from checkpoint.

        Args:
            checkpoint_path: Path to .ckpt file (e.g. checkpoints/re10k.ckpt).
            config_overrides: Optional hydra-style overrides dict.
            mvsplat_root: Path to MVSplat repo root.  If None, uses MVSPLAT_ROOT
                env var or local baselines/mvsplat/ fallback.
        """
        root = _resolve_mvsplat_root(mvsplat_root)
        previous_cwd, saved_path, saved_modules = _ensure_mvsplat_on_path(str(root))

        import hydra
        from hydra.core.global_hydra import GlobalHydra

        # Clear Hydra singleton if already initialized (VGGT or others may have used it).
        with contextlib.suppress(Exception):
            GlobalHydra.instance().clear()

        from src.config import load_typed_root_config
        from src.global_cfg import set_cfg
        from src.misc.step_tracker import StepTracker
        from src.model.decoder import get_decoder
        from src.model.encoder import get_encoder
        from src.model.model_wrapper import ModelWrapper

        # Build config via hydra compose.
        hydra_overrides = [
            "+experiment=re10k",
            f"checkpointing.load={checkpoint_path}",
            "mode=test",
            "dataset/view_sampler=evaluation",
            "++dataset.view_sampler.index_path=assets/evaluation_index_re10k.json",
            "dataset.skip_bad_shape=false",
            "test.compute_scores=false",
            "test.save_image=false",
            "test.save_video=false",
        ]
        # Apply caller overrides (merged after defaults so caller can override).
        if config_overrides:
            for key, value in config_overrides.items():
                hydra_overrides.append(f"{key}={value}")

        with hydra.initialize_config_dir(
            config_dir=str(root / "config"),
            version_base=None,
        ):
            cfg_dict = hydra.compose(
                config_name="main",
                overrides=hydra_overrides,
            )

        cfg = load_typed_root_config(cfg_dict)
        set_cfg(cfg_dict)
        self._cfg = cfg_dict

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
        self._model = model
        self._loaded_checkpoint = checkpoint_path

        print(
            f"[MVSplatWrapper] Loaded checkpoint: {checkpoint_path} "
            f"(GPU: {torch.cuda.get_device_name(0)})"
        )
        # Restore original paths, cwd, and modules.
        for key in list(sys.modules.keys()):
            if key == "src" or key.startswith("src.") or key.startswith("config") or key.startswith("dataset") or key.startswith("model") or key.startswith("loss") or key.startswith("misc") or key.startswith("evaluation") or key.startswith("global_cfg"):
                del sys.modules[key]
        sys.path[:] = saved_path
        sys.modules.update(saved_modules)
        os.chdir(previous_cwd)

    # ---- infer ------------------------------------------------------------

    def infer(
        self,
        images: torch.Tensor,
        poses: torch.Tensor,
        K: torch.Tensor,
    ) -> dict:
        """Extract Gaussians from input views.

        Args:
            images: (B, V, 3, H, W) float32, RGB in [0, 1].
            poses:  (B, V, 4, 4) float32, world-from-camera matrices.
            K:      (B, V, 3, 3) float32, camera intrinsics.

        Returns:
            Dict with:
              - gaussians: GaussianData (means, opacities, scales, covariances, ...)
              - metadata: dict with occupancy_alpha placeholder.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call build() first.")

        b, v, c, h, w = images.shape
        device = images.device

        # Build a minimal batch dict compatible with MVSplat encoder.
        # The encoder expects: context -> image, extrinsics, intrinsics, near, far, index.
        batch_context = {
            "image": images,  # (B, V, C, H, W)
            "extrinsics": poses.float(),
            "intrinsics": K.float(),
            "near": torch.full((b, v), 0.2, device=device),  # placeholder
            "far": torch.full((b, v), 100.0, device=device),  # placeholder
            "index": torch.arange(v, device=device)[None, :].expand(b, -1),
        }

        with torch.no_grad():
            gaussians = self._model.encoder(
                batch_context, global_step=0, deterministic=False
            )

        # Extract to numpy (first batch element).
        means = gaussians.means[0].cpu().numpy()
        opacities = gaussians.opacities[0].cpu().numpy()
        covariances = gaussians.covariances[0].cpu().numpy()

        has_scales = hasattr(gaussians, "scales") and gaussians.scales is not None
        scales = (
            gaussians.scales[0].cpu().numpy()
            if has_scales
            else np.full((means.shape[0], 3), 0.05, dtype=np.float32)
        )

        has_harmonics = hasattr(gaussians, "harmonics") and gaussians.harmonics is not None
        harmonics = (
            gaussians.harmonics[0].cpu().numpy()
            if has_harmonics
            else np.zeros((means.shape[0], 3, 1), dtype=np.float32)
        )

        has_rotations = hasattr(gaussians, "rotations") and gaussians.rotations is not None
        rotations = gaussians.rotations[0].cpu().numpy() if has_rotations else None

        g_data = GaussianData(
            means=means,
            opacities=opacities,
            scales=scales,
            covariances=covariances,
            harmonics=harmonics,
            rotations=rotations,
        )

        n = means.shape[0]
        print(f"[MVSplatWrapper] Extracted {n} Gaussians (B={b}, V={v}, H={h}, W={w}).")

        # Free GPU memory used by the raw Gaussian tensors before returning.
        del gaussians
        torch.cuda.empty_cache()

        return {
            "gaussians": g_data,
            "metadata": {
                "occupancy_alpha": opacities,  # placeholder; real alpha needs training
                "n_gaussians": n,
                "has_scales": has_scales,
                "has_rotations": has_rotations,
            },
        }

    # ---- occupancy extraction ---------------------------------------------

    def extract_occupancy(
        self,
        gaussians: GaussianData,
        config: dict | None = None,
    ) -> dict:
        """Compute planning-oriented occupancy from Gaussian parameters.

        Args:
            gaussians: GaussianData from infer().
            config: Dict with keys:
                height_filter: [y_min, y_max] in Y-up coordinates (default [-0.5, 2.0]).
                alpha_threshold: minimum opacity for occupancy (default 0.3).

        Returns:
            Dict with:
              - occupancy_alpha: (N,) per-Gaussian occupancy score.
              - bev_footprint: (H, W) float32 BEV occupancy grid.
              - bbox_3d: (center_x, center_y, center_z, size_x, size_y, size_z) or None.
        """
        cfg = config or {}
        height_range = cfg.get("height_filter", (-0.5, 2.0))
        alpha_threshold = cfg.get("alpha_threshold", 0.3)
        grid_res = cfg.get("bev_resolution", 0.05)
        grid_half = cfg.get("bev_range", [-10, -10, 10, 10])
        # Use x and z for BEV in Y-up.
        x_min, z_min, x_max, z_max = grid_half[0], grid_half[1], grid_half[2], grid_half[3]

        means = gaussians.means
        opacities = gaussians.opacities
        scales = gaussians.scales

        # Height filter (Y-up: y is height).
        mask = (means[:, 1] >= height_range[0]) & (means[:, 1] <= height_range[1])
        m = means[mask]
        o = opacities[mask]
        s = scales[mask]

        n_cells_x = int((x_max - x_min) / grid_res)
        n_cells_z = int((z_max - z_min) / grid_res)
        bev = np.zeros((n_cells_z, n_cells_x), dtype=np.float32)

        for i in range(len(m)):
            x, y, z = m[i]
            sx, sy, sz = s[i]
            col = int((x - x_min) / grid_res)
            row = int((z - z_min) / grid_res)
            if 0 <= row < n_cells_z and 0 <= col < n_cells_x:
                # footprint radius from XZ scale.
                r = max(sx, sz) * 3.0  # 3-sigma
                rc = max(1, int(r / grid_res))
                for dr in range(-rc, rc + 1):
                    for dc in range(-rc, rc + 1):
                        nr, nc = row + dr, col + dc
                        if 0 <= nr < n_cells_z and 0 <= nc < n_cells_x:
                            dist = np.sqrt(dr**2 + dc**2) * grid_res
                            if dist <= r:
                                w = o[i] * np.exp(-0.5 * (dist / max(r, 1e-6)) ** 2)
                                bev[nr, nc] = max(bev[nr, nc], w)

        # BBox 3D: crude bounding box from valid points.
        if len(m) > 0:
            bbox_3d = (
                float(m[:, 0].mean()),
                float(m[:, 1].mean()),
                float(m[:, 2].mean()),
                float(m[:, 0].max() - m[:, 0].min()),
                float(m[:, 1].max() - m[:, 1].min()),
                float(m[:, 2].max() - m[:, 2].min()),
            )
        else:
            bbox_3d = None

        print(
            f"[MVSplatWrapper] Occupancy: {len(m)} Gaussians in height range, "
            f"BEV grid {bev.shape}, "
            f"occupied cells (>{alpha_threshold}): {(bev > alpha_threshold).sum()}"
        )

        return {
            "occupancy_alpha": o,
            "bev_footprint": bev,
            "bbox_3d": bbox_3d,
        }

    # ---- identity encoding (Gaussian Grouping style) ---------------------

    def add_identity_encoding(
        self,
        gaussians: GaussianData,
        dim: int = 16,
    ) -> GaussianData:
        """Add per-Gaussian identity encoding for semantic lifting.

        Initialises a (N, dim) learnable embedding per Gaussian, modelled after
        Gaussian Grouping's _objects_dc.  The encoding is trained via
        differentiable rendering + classification head + 2D mask supervision.

        Call train_identity_encoding() to optimise the embeddings.

        Args:
            gaussians: GaussianData from infer().
            dim: Dimensionality of identity embedding (default 16).

        Returns:
            Same GaussianData with identity_encoding set.
        """
        n = len(gaussians.means)
        # Init near zero (small random values) so initial classification is uniform.
        gaussians.identity_encoding = np.random.randn(n, dim).astype(np.float32) * 0.05
        print(f"[MVSplatWrapper] Initialised identity encoding: ({n}, {dim})")
        return gaussians

    def train_identity_encoding(
        self,
        gaussians: GaussianData,
        image: np.ndarray,
        mask_2d: np.ndarray,  # (H, W) int32, 0=background, 1..K=object classes
        camera_pose: np.ndarray,  # (4, 4) world-from-camera
        K: np.ndarray,  # (3, 3) intrinsics
        num_classes: int | None = None,
        iterations: int = 1000,
        lr: float = 5e-3,
        device: str = "cuda",
    ) -> GaussianData:
        """Train per-Gaussian identity encoding via 2D mask supervision.

        Uses differentiable Gaussian rasterization to render the identity
        encoding to 2D, then a 1×1 conv classifier maps (D_id, H, W) →
        (K, H, W) logits, supervised by cross-entropy against the 2D mask.

        This is the feedforward analog of Gaussian Grouping's training loop:
        freeze geometry (means, scales, rotations, opacities), train only
        identity encoding + classifier head.

        Args:
            gaussians: GaussianData with identity_encoding initialised.
            image: (H, W, 3) RGB image for reference (not used in loss).
            mask_2d: (H, W) int32, 0=background, 1..K per-object class labels.
            camera_pose: (4, 4) world-from-camera (C2W), Y-up.
            K: (3, 3) camera intrinsics.
            num_classes: Number of semantic classes. Auto-detect if None.
            iterations: Training iterations (default 1000, ~30s on A6000).
            lr: Learning rate for identity encoding + classifier.
            device: "cuda" or "cpu".

        Returns:
            Same GaussianData with trained identity_encoding.
        """
        if gaussians.identity_encoding is None:
            raise ValueError("Call add_identity_encoding() first.")

        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        device_t = torch.device(device if torch.cuda.is_available() else "cpu")

        k = num_classes if num_classes is not None else int(mask_2d.max()) + 1
        d_id = gaussians.identity_encoding.shape[1]

        # ---- Move Gaussian params to GPU tensors ----
        means_t = torch.from_numpy(gaussians.means).float().to(device_t)
        scales_t = torch.from_numpy(gaussians.scales).float().to(device_t)
        opacities_t = torch.from_numpy(gaussians.opacities).float().to(device_t)
        rotations_t = torch.from_numpy(gaussians.rotations).float().to(device_t) if gaussians.rotations is not None else None

        # Identity encoding: (N, D_id) → need SH-like format (D_id, N, 1) for rasterizer.
        id_enc = torch.from_numpy(gaussians.identity_encoding).float().to(device_t)
        id_enc_param = nn.Parameter(id_enc.unsqueeze(-1).permute(1, 0, 2).contiguous())
        # (D_id, N, 1)

        # Classifier: 1×1 conv (D_id → K).
        classifier = nn.Conv2d(d_id, k, kernel_size=1).to(device_t)

        # Camera params.
        w2c = torch.inverse(torch.from_numpy(camera_pose).float().to(device_t))  # world→cam
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        H, W = mask_2d.shape

        # 2D mask tensor.
        mask_t = torch.from_numpy(mask_2d.astype(np.int64)).long().to(device_t)

        # Optimizer.
        opt = torch.optim.Adam([
            {"params": [id_enc_param], "lr": lr},
            {"params": classifier.parameters(), "lr": lr},
        ])

        # ---- Training loop ----
        print(f"[IdentityTraining] Training {iterations} iters, {k} classes, {d_id}-dim identity...")
        for it in range(iterations):
            opt.zero_grad()

            # --- Simplified differentiable Gaussian rasterization ---
            # Project 3D means to 2D using pinhole camera.
            means_cam = (w2c[:3, :3] @ means_t.T + w2c[:3, 3:4]).T  # (N, 3) in cam space
            z = means_cam[:, 2].clamp(min=1e-4)
            u = fx * means_cam[:, 0] / z + cx
            v = fy * means_cam[:, 1] / z + cy

            # Filter Gaussians outside image.
            valid = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0.01)
            if valid.sum() < 10:
                continue

            u_v = u[valid].long()
            v_v = v[valid].long()
            z_v = z[valid]
            alpha_v = torch.sigmoid(opacities_t[valid])
            # Scale to pixel-space radius (approximate).
            radius_v = torch.clamp(scales_t[valid, :2].norm(dim=1) * fx / z_v, min=1.0, max=20.0)

            # Identity features for valid Gaussians.
            id_v = id_enc_param[:, valid, 0]  # (D_id, M)

            # --- Splat identity to 2D grid (simplified: weighted sum per pixel) ---
            rendered_id = torch.zeros(d_id, H, W, device=device_t)
            weight_sum = torch.zeros(H, W, device=device_t)

            sigma = radius_v / 2.0  # Gaussian sigma in pixels
            for gi in range(valid.sum().item()):
                r = int(radius_v[gi].item())
                r = min(r, 30)  # limit splat radius for speed
                if r < 1:
                    continue
                ui, vi = u_v[gi].item(), v_v[gi].item()
                s = sigma[gi].item()
                # Local window around (ui, vi).
                umin, umax = max(0, ui - r), min(W, ui + r + 1)
                vmin, vmax = max(0, vi - r), min(H, vi + r + 1)
                du = torch.arange(umin, umax, device=device_t).float() - ui
                dv = torch.arange(vmin, vmax, device=device_t).float() - vi
                du_grid, dv_grid = torch.meshgrid(du, dv, indexing="xy")
                dist2 = du_grid**2 + dv_grid**2
                weight = alpha_v[gi] * torch.exp(-0.5 * dist2 / (s**2 + 1e-6))
                # Accumulate weighted identity.
                for ch in range(d_id):
                    rendered_id[ch, vmin:vmax, umin:umax] += weight * id_v[ch, gi]
                weight_sum[vmin:vmax, umin:umax] += weight

            # Normalize.
            weight_sum = weight_sum.clamp(min=1e-6)
            rendered_id = rendered_id / weight_sum.unsqueeze(0)

            # Classification.
            logits = classifier(rendered_id.unsqueeze(0))[0]  # (K, H, W)
            loss = F.cross_entropy(logits.unsqueeze(0), mask_t.unsqueeze(0))

            loss.backward()
            opt.step()

            if it % 200 == 0:
                pred = logits.argmax(dim=0)
                acc = (pred == mask_t).float().mean().item()
                print(f"  [IdentityTraining] iter {it:4d}/{iterations}  loss={loss.item():.4f}  acc={acc:.4f}")

        # ---- Extract trained identity ----
        with torch.no_grad():
            trained_id = id_enc_param.detach().permute(1, 0, 2).squeeze(-1).cpu().numpy()  # (N, D_id)
        gaussians.identity_encoding = trained_id.astype(np.float32)

        # ---- Predict per-Gaussian object labels ----
        # Use the classifier to predict labels from 3D identity encoding.
        # classifier expects (B, D_id, H, W) — treat N Gaussians as spatial dim.
        with torch.no_grad():
            id_3d = id_enc_param.detach().permute(2, 0, 1)  # (N, D_id, 1)
            id_3d_in = id_3d.unsqueeze(-1).permute(1, 0, 2, 3)  # (D_id, N, 1, 1) → need (B, D_id, N, 1)
            id_3d_in = id_3d_in.permute(0, 1, 2, 3)  # already (D_id, N, 1, 1), just need B dim
            # Correct: (1, D_id, N, 1)
            id_3d_batch = id_3d.unsqueeze(0).permute(0, 2, 1, 3)  # (1, D_id, N, 1)
            logits_3d = classifier(id_3d_batch)  # (1, K, N, 1)
            per_gaussian_labels = logits_3d[0, :, :, 0].argmax(dim=0).cpu().numpy()  # (N,)

        print(f"[IdentityTraining] Done. Per-Gaussian labels: {len(set(per_gaussian_labels)) - 1} objects detected.")
        return gaussians, per_gaussian_labels

    # ---- checkpoint mgmt --------------------------------------------------

    def save(self, ckpt_path: str) -> None:
        """Save wrapper state (for fine-tuning scenarios)."""
        if self._model is None:
            return
        torch.save(
            {
                "model_state": self._model.state_dict(),
                "checkpoint": self._loaded_checkpoint,
            },
            ckpt_path,
        )

    def load(self, ckpt_path: str) -> None:
        """Load fine-tuned wrapper state.

        Note: This is for loading *already-built* wrapper checkpoints, not MVSplat ckpts.
        For fresh model loading use build().
        """
        data = torch.load(ckpt_path, map_location="cpu")
        if self._model is not None:
            self._model.load_state_dict(data["model_state"])
        self._loaded_checkpoint = data.get("checkpoint", ckpt_path)

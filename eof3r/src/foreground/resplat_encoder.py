"""Lightweight ReSplat encoder loader for Phase B training.

Loads the ReSplat encoder directly from a checkpoint file, bypassing
the full Hydra config system. This avoids config version mismatches.

Usage:
    encoder = load_resplat_encoder("pretrained/resplat-base-re10k-256x256-view2.pth")
    output = encoder(context, step=0, deterministic=True)
    gaussians = output["gaussians"]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_RESPLAT_ROOT = Path(__file__).parents[3] / "baselines" / "resplat"


def load_resplat_encoder(
    checkpoint_path: str,
    device: str = "cuda",
) -> nn.Module:
    """Load ReSplat encoder from checkpoint with automatic config detection.

    Instead of matching configs, this function:
    1. Creates a minimal encoder with the right architecture
    2. Loads weights with shape matching
    3. Returns a callable encoder

    Args:
        checkpoint_path: Path to ReSplat .pth checkpoint.
        device: Device to load on.

    Returns:
        Encoder module that takes (context, step, deterministic) and returns
        dict with "gaussians" key.
    """
    # Add ReSplat to path
    if str(_RESPLAT_ROOT) not in sys.path:
        sys.path.insert(0, str(_RESPLAT_ROOT))

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)

    # Separate encoder weights
    encoder_sd = {}
    for k, v in sd.items():
        if k.startswith("encoder."):
            encoder_sd[k.replace("encoder.", "")] = v

    if not encoder_sd:
        raise ValueError("No encoder weights found in checkpoint")

    # Detect dimensions from checkpoint
    proj_weight = encoder_sd.get("proj.weight")
    gh_weight = encoder_sd.get("gaussian_head.0.weight")
    gr_weight = encoder_sd.get("gaussian_regressor.0.weight")

    if proj_weight is None or gh_weight is None:
        raise ValueError("Cannot detect encoder architecture from checkpoint")

    proj_in = proj_weight.shape[1]  # e.g., 797
    gh_out = gh_weight.shape[0]     # e.g., 232
    gr_in = gr_weight.shape[1] if gr_weight is not None else None

    logger.info(
        f"ReSplat checkpoint: proj_in={proj_in}, gh_out={gh_out}, "
        f"gr_in={gr_in}, gaussians_per_point={gh_out // 58 if gh_out % 58 == 0 else '?'}"
    )

    # Build encoder using ReSplat's factory with auto-detected config
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from src.model.encoder import get_encoder

    config_dir = str(_RESPLAT_ROOT / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="main", overrides=["+experiment=re10k"])

    # Override config to match checkpoint dimensions
    enc_cfg = cfg.model.encoder

    # Detect init_gaussian_multiple from output dimension
    # Standard: 58 params per Gaussian (3 mean + 3 scale + 4 rot + 1 opacity + 47 SH)
    params_per_gaussian = 58
    init_gaussian_multiple = gh_out // params_per_gaussian
    if gh_out % params_per_gaussian != 0:
        # Try other possibilities
        for mult in [1, 2, 4, 8]:
            if gh_out % mult == 0:
                init_gaussian_multiple = mult
                params_per_gaussian = gh_out // mult
                break

    enc_cfg.init_gaussian_multiple = init_gaussian_multiple

    # We need to figure out the feature_upsampler_channels that gives proj_in.
    # This depends on monodepth_vit_type, latent_downsample, and fixed_latent_size.
    # Try different combinations.
    model_configs = {
        'vits': 384, 'vitb': 768, 'vitl': 1024,
    }

    found = False
    for vit_type, vit_channels in model_configs.items():
        for latent_ds in [4, 2, 8]:
            for fixed_latent in [True, False]:
                enc_cfg.monodepth_vit_type = vit_type
                enc_cfg.latent_downsample = latent_ds
                enc_cfg.fixed_latent_size = fixed_latent

                # Calculate feature_upsampler_channels
                if latent_ds == 4:
                    feat = vit_channels // 4 + 128 + 64 + 96 + 128
                elif latent_ds == 2:
                    feat = vit_channels // 64 * 4 + 128 // 4 + 64 + 96 + 128 // 4
                elif latent_ds == 8:
                    if fixed_latent:
                        feat = vit_channels // 4 + 128 + 64 + 96 + 128
                    else:
                        feat = vit_channels + 128 + 64 + 96 + 128

                # Calculate proj input dimension
                # proj_in = 3 + feat + gaussian_regressor_channels + 1
                # With fixed_latent_size: proj_in = (3 + feat + 512 + 1) - 3 + 48
                gr_channels = enc_cfg.gaussian_regressor_channels
                if fixed_latent:
                    calc_proj_in = 3 + feat + gr_channels + 1 - 3 + 3 * (4 ** 2)
                else:
                    calc_proj_in = 3 + feat + gr_channels + 1 - 3 + 3 * (latent_ds ** 2)

                if calc_proj_in == proj_in:
                    found = True
                    logger.info(
                        f"Matched config: vit={vit_type}, latent_ds={latent_ds}, "
                        f"fixed_latent={fixed_latent}, feat={feat}"
                    )
                    break
            if found:
                break
        if found:
            break

    if not found:
        logger.warning(
            f"Could not match proj_in={proj_in} exactly. "
            f"Using vitb/latent_ds=4/fixed_latent=True as fallback."
        )
        enc_cfg.monodepth_vit_type = "vitb"
        enc_cfg.latent_downsample = 4
        enc_cfg.fixed_latent_size = True

    # Create encoder
    encoder, _ = get_encoder(enc_cfg)

    # Load weights with flexible matching
    model_sd = encoder.state_dict()
    loaded = {}
    skipped = []
    for k, v in encoder_sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            loaded[k] = v
        else:
            skipped.append(k)

    encoder.load_state_dict(loaded, strict=False)
    logger.info(f"Loaded {len(loaded)} tensors, skipped {len(skipped)} (shape mismatch)")

    encoder = encoder.to(device).eval()
    return encoder

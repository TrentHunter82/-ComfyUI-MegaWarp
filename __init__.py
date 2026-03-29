"""
ComfyUI-MegaWarp -- Modern video stylization with MegaFlow optical flow.

Replaces WarpFusion's RAFT-based pipeline with:
  - MegaFlow: zero-shot large displacement optical flow via frozen
    DINO ViT features and multi-frame temporal attention
  - Latent-space warping to avoid VAE feedback loop
  - Soft confidence masks instead of binary consistency checks
  - Compatible with any ComfyUI diffusion backend

@author  Trent (flippingsigmas)
@title   MegaWarp
@license Apache-2.0
"""

import sys
import importlib

# -- Version gate ---------------------------------------------------
MIN_PYTHON = (3, 12)
MIN_TORCH = (2, 7)

if sys.version_info < MIN_PYTHON:
    raise RuntimeError(
        f"ComfyUI-MegaWarp requires Python "
        f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}+. "
        f"You have {sys.version_info.major}."
        f"{sys.version_info.minor}."
    )

try:
    import torch
    _tv = tuple(
        int(x) for x in torch.__version__.split(".")[:2]
    )
    if _tv < MIN_TORCH:
        raise RuntimeError(
            f"ComfyUI-MegaWarp requires PyTorch "
            f"{MIN_TORCH[0]}.{MIN_TORCH[1]}+. "
            f"You have {torch.__version__}."
        )
except ImportError:
    raise RuntimeError(
        "ComfyUI-MegaWarp requires PyTorch. "
        "Please install it first."
    )

# -- Node registration via per-module mappings ----------------------

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}

_MODULES = [
    ".nodes.model_loader",
    ".nodes.flow_estimate",
    ".nodes.flow_warp",
    ".nodes.latent_warp",
    ".nodes.flow_consistency",
    ".nodes.flow_blend",
    ".nodes.flow_condition_warp",
    ".nodes.flow_noise_warp",
    ".nodes.point_tracker",
    ".nodes.flow_to_mask",
    ".nodes.flow_visualize",
]

for _mod_path in _MODULES:
    try:
        _mod = importlib.import_module(_mod_path, package=__package__)
        NODE_CLASS_MAPPINGS.update(
            getattr(_mod, "NODE_CLASS_MAPPINGS", {})
        )
        NODE_DISPLAY_NAME_MAPPINGS.update(
            getattr(_mod, "NODE_DISPLAY_NAME_MAPPINGS", {})
        )
    except Exception as _exc:
        print(
            f"[MegaWarp] Skipping {_mod_path}: {_exc}"
        )

WEB_DIRECTORY = "./js"
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

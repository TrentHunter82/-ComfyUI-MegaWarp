"""
MegaFlow Model Loader -- Download and cache the MegaFlow
optical flow checkpoint from HuggingFace.
"""

import torch
from typing import Any, Dict, Tuple

import comfy.model_management as mm


class MegaFlowModelLoader:
    """
    Load a MegaFlow model for optical flow estimation or
    point tracking. Models are cached in memory to avoid
    redundant downloads and GPU transfers.
    """

    CATEGORY = "MegaWarp/Models"
    RETURN_TYPES = ("MEGAFLOW_MODEL",)
    RETURN_NAMES = ("model",)
    OUTPUT_TOOLTIPS = (
        "MegaFlow model ready for flow estimation "
        "or point tracking",
    )
    FUNCTION = "load_model"
    DESCRIPTION = (
        "Load a MegaFlow optical flow model from "
        "HuggingFace. Supports flow estimation "
        "(megaflow-flow, megaflow-chairs-things) and "
        "point tracking (megaflow-track). Models are "
        "cached after first load."
    )

    _cached_model = None
    _cached_name = None
    _cached_device = None

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model_name": (
                    [
                        "megaflow-flow",
                        "megaflow-chairs-things",
                        "megaflow-track",
                    ],
                    {"default": "megaflow-flow"},
                ),
            },
        }

    def load_model(
        self,
        model_name: str,
    ) -> Tuple[Any]:
        device = mm.get_torch_device()
        cached = MegaFlowModelLoader
        if (
            cached._cached_model is not None
            and cached._cached_name == model_name
            and cached._cached_device == str(device)
        ):
            return (cached._cached_model,)

        try:
            from megaflow import MegaFlow
        except ImportError:
            raise ImportError(
                "MegaFlow is not installed. Run:\n"
                "  pip install "
                "git+https://github.com/cvg/"
                "megaflow.git"
                "\nRequires Python >= 3.12 and "
                "PyTorch >= 2.7"
            )

        print(
            f"[MegaWarp] Loading MegaFlow model: "
            f"{model_name}"
        )
        mm.unload_all_models()
        mm.soft_empty_cache()

        model = (
            MegaFlow.from_pretrained(
                model_name, device="cpu"
            )
            .eval()
            .to(device)
        )
        mm.soft_empty_cache()

        cached._cached_model = model
        cached._cached_name = model_name
        cached._cached_device = str(device)

        print(
            f"[MegaWarp] MegaFlow model loaded on "
            f"{device}"
        )
        return (model,)


NODE_CLASS_MAPPINGS = {
    "MegaFlowModelLoader": MegaFlowModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MegaFlowModelLoader": "MegaFlow Model Loader",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

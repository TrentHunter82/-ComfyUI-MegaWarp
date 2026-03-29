"""
MegaFlow Latent Warp -- Warp latent-space tensors using
optical flow, skipping the VAE encode/decode cycle to avoid
cumulative quantization error.
"""

import torch
from typing import Any, Dict, Tuple

from ..utils.warp_ops import (
    warp_with_flow,
    scale_flow_to_resolution,
)
from ..utils.tensor_utils import (
    select_flow_pair,
    flow_field_to_comfyui_mask,
)


class MegaFlowLatentWarp:
    """
    Warp diffusion latents directly using optical flow,
    bypassing the VAE encode/decode loop that causes color
    drift in WarpFusion. Flow is downscaled to latent
    resolution with proper magnitude compensation.
    """

    CATEGORY = "MegaWarp/Warp"
    RETURN_TYPES = ("LATENT", "MASK")
    RETURN_NAMES = ("warped_latent", "valid_mask")
    OUTPUT_TOOLTIPS = (
        "Warped latent samples at 1/8 resolution",
        "Valid pixel mask showing warped coverage",
    )
    FUNCTION = "warp_latent"
    DESCRIPTION = (
        "Warp latent-space tensors using optical flow. "
        "Skips the VAE encode/decode cycle per frame, "
        "avoiding WarpFusion's cumulative color drift. "
        "Flow is automatically downscaled to latent "
        "resolution with magnitude compensation."
    )

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "latent": ("LATENT",),
                "flow": ("FLOW_FIELD",),
            },
            "optional": {
                "frame_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                    "tooltip": (
                        "Which flow pair to use. "
                        "0 = flow from frame 0 to 1."
                    ),
                }),
                "flow_scale_mode": (
                    ["area", "bilinear"],
                    {
                        "default": "area",
                        "tooltip": (
                            "Interpolation mode for "
                            "downscaling flow to latent "
                            "resolution"
                        ),
                    },
                ),
            },
        }

    def warp_latent(
        self,
        latent: dict,
        flow: torch.Tensor,
        frame_index: int = 0,
        flow_scale_mode: str = "area",
    ) -> Tuple[dict, torch.Tensor]:
        samples = latent["samples"]  # [B, 4, Hl, Wl]
        _, _, lat_h, lat_w = samples.shape

        if flow.shape[0] > 1:
            idx = min(frame_index, flow.shape[0] - 1)
            single_flow = select_flow_pair(flow, idx)
        else:
            single_flow = flow

        single_flow = single_flow.to(samples.device)

        # Downscale flow to latent resolution with
        # magnitude compensation
        scaled_flow = scale_flow_to_resolution(
            single_flow,
            lat_h,
            lat_w,
            mode=flow_scale_mode,
        )

        if (
            scaled_flow.shape[0] == 1
            and samples.shape[0] > 1
        ):
            scaled_flow = scaled_flow.expand(
                samples.shape[0], -1, -1, -1
            )

        warped, valid = warp_with_flow(
            samples, scaled_flow, mode="bilinear"
        )

        # Preserve all LATENT dict keys
        result = latent.copy()
        result["samples"] = warped

        return (
            result,
            flow_field_to_comfyui_mask(valid),
        )


NODE_CLASS_MAPPINGS = {
    "MegaFlowLatentWarp": MegaFlowLatentWarp,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MegaFlowLatentWarp": "MegaFlow Latent Warp",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

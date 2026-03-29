"""
Flow Noise Warp -- Warp latent noise using optical flow for
temporally coherent diffusion. Based on 'How I Warped Your
Noise' (arXiv:2410.16152).
"""

import torch
from typing import Any, Dict, Tuple

from ..utils.warp_ops import (
    warp_with_flow,
    scale_flow_to_resolution,
)
from ..utils.tensor_utils import select_flow_pair


class FlowNoiseWarp:
    """
    Warp the previous frame's latent noise by optical flow
    so the diffusion model starts from a spatially
    consistent noise field. Disoccluded regions receive
    fresh random noise. Complements latent warping for
    maximum temporal consistency.
    """

    CATEGORY = "MegaWarp/Advanced"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("warped_noise",)
    OUTPUT_TOOLTIPS = (
        "Flow-warped latent noise for temporally "
        "coherent diffusion initialization",
    )
    FUNCTION = "warp_noise"
    DESCRIPTION = (
        "Warp latent noise using optical flow instead "
        "of starting each frame from random noise. "
        "Produces temporally consistent texture "
        "generation. Blend controls the mix of warped "
        "vs fresh noise; disoccluded regions always "
        "get fresh noise."
    )

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "latent": ("LATENT",),
                "flow": ("FLOW_FIELD",),
            },
            "optional": {
                "noise_blend": ("FLOAT", {
                    "default": 0.85,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": (
                        "1.0 = fully warped noise, "
                        "0.0 = fresh random noise"
                    ),
                }),
                "frame_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                    "tooltip": (
                        "Which flow pair to use. "
                        "0 = flow from frame 0 to 1."
                    ),
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": (
                        "Seed for fresh noise in "
                        "disoccluded regions"
                    ),
                }),
            },
        }

    def warp_noise(
        self,
        latent: dict,
        flow: torch.Tensor,
        noise_blend: float = 0.85,
        frame_index: int = 0,
        seed: int = 0,
    ) -> Tuple[dict]:
        samples = latent["samples"]  # [B, 4, Hl, Wl]
        _, _, lat_h, lat_w = samples.shape

        if flow.shape[0] > 1:
            idx = min(frame_index, flow.shape[0] - 1)
            single_flow = select_flow_pair(flow, idx)
        else:
            single_flow = flow

        single_flow = single_flow.to(samples.device)

        # Downscale flow to latent resolution
        scaled_flow = scale_flow_to_resolution(
            single_flow, lat_h, lat_w, mode="area"
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

        # Generate seeded fresh noise
        gen = torch.Generator(
            device=samples.device
        ).manual_seed(seed)
        fresh = torch.randn(
            samples.shape,
            device=samples.device,
            dtype=samples.dtype,
            generator=gen,
        )

        # Blend warped and fresh noise
        result = (
            noise_blend * warped
            + (1.0 - noise_blend) * fresh
        )

        # Force fresh noise in disoccluded regions
        invalid = (valid < 0.5).expand_as(result)
        result[invalid] = fresh[invalid]

        output = latent.copy()
        output["samples"] = result

        return (output,)


NODE_CLASS_MAPPINGS = {
    "FlowNoiseWarp": FlowNoiseWarp,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlowNoiseWarp": "Flow Noise Warp",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

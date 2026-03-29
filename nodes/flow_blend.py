"""
Flow Blend -- Confidence-weighted composite of warped
stylized frames and original video frames.
"""

import torch
import torch.nn.functional as F
from typing import Any, Dict, Tuple

from ..utils.tensor_utils import (
    comfyui_mask_to_4d,
    comfyui_to_image_chw,
    image_chw_to_comfyui,
)


class FlowBlend:
    """
    Blend a flow-warped stylized frame with the original
    video frame using a soft confidence mask. High
    confidence regions keep the warped style; low
    confidence regions fall back to the original.
    """

    CATEGORY = "MegaWarp/Composite"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("blended",)
    OUTPUT_TOOLTIPS = (
        "Confidence-weighted blend of warped and "
        "original frames",
    )
    FUNCTION = "blend"
    DESCRIPTION = (
        "Spatially-adaptive blend of a flow-warped "
        "stylized frame with the original video frame. "
        "Uses the confidence mask to favor the warped "
        "result in reliable regions and the original in "
        "occluded areas. Optional bias and mask blur."
    )

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "warped": ("IMAGE",),
                "original": ("IMAGE",),
                "confidence": ("MASK",),
            },
            "optional": {
                "blend_bias": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": (
                        "+bias favors warped stylized "
                        "frame, -bias favors original"
                    ),
                }),
                "mask_blur": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 32,
                    "step": 1,
                    "tooltip": (
                        "Box blur radius on confidence "
                        "mask for smoother transitions"
                    ),
                }),
            },
        }

    def blend(
        self,
        warped: torch.Tensor,
        original: torch.Tensor,
        confidence: torch.Tensor,
        blend_bias: float = 0.0,
        mask_blur: int = 0,
    ) -> Tuple[torch.Tensor]:
        alpha = comfyui_mask_to_4d(confidence)
        alpha = alpha.to(warped.device)

        alpha = torch.clamp(alpha + blend_bias, 0.0, 1.0)

        if mask_blur > 0:
            k = mask_blur * 2 + 1
            kernel = torch.ones(
                1, 1, k, k, device=alpha.device
            ) / (k * k)
            alpha_padded = F.pad(
                alpha, [mask_blur] * 4, mode="replicate"
            )
            alpha = F.conv2d(alpha_padded, kernel)

        w_chw = comfyui_to_image_chw(warped)
        o_chw = comfyui_to_image_chw(original)

        if alpha.shape[2:] != w_chw.shape[2:]:
            alpha = F.interpolate(
                alpha,
                size=w_chw.shape[2:],
                mode="bilinear",
            )

        alpha = alpha.expand_as(w_chw)
        blended = alpha * w_chw + (1.0 - alpha) * o_chw

        return (image_chw_to_comfyui(blended),)


NODE_CLASS_MAPPINGS = {
    "FlowBlend": FlowBlend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlowBlend": "Flow Blend",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

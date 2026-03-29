"""
MegaFlow Warp -- Warp an image using a dense optical flow
field in pixel space.
"""

import torch
from typing import Any, Dict, Tuple

from ..utils.warp_ops import (
    warp_with_flow,
    scale_flow_to_resolution,
)
from ..utils.tensor_utils import (
    comfyui_to_image_chw,
    image_chw_to_comfyui,
    select_flow_pair,
    flow_field_to_comfyui_mask,
)


class MegaFlowWarp:
    """
    Warp a single image frame according to a pre-computed
    optical flow field. Uses GPU-accelerated grid_sample
    for fast bilinear or bicubic interpolation.
    """

    CATEGORY = "MegaWarp/Warp"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("warped_image", "valid_mask")
    OUTPUT_TOOLTIPS = (
        "Image warped according to the flow field",
        "Mask showing which pixels have valid source "
        "data after warping",
    )
    FUNCTION = "warp"
    DESCRIPTION = (
        "Warp an image in pixel space using optical "
        "flow. Produces a warped image and a valid "
        "mask showing coverage. Connect flow from "
        "MegaFlow Estimate."
    )

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
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
                "interpolation": (
                    ["bilinear", "bicubic"],
                    {"default": "bilinear"},
                ),
            },
        }

    def warp(
        self,
        image: torch.Tensor,
        flow: torch.Tensor,
        frame_index: int = 0,
        interpolation: str = "bilinear",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_chw = comfyui_to_image_chw(image)

        if flow.shape[0] > 1:
            idx = min(frame_index, flow.shape[0] - 1)
            single_flow = select_flow_pair(flow, idx)
        else:
            single_flow = flow

        single_flow = single_flow.to(img_chw.device)

        if single_flow.shape[2:] != img_chw.shape[2:]:
            single_flow = scale_flow_to_resolution(
                single_flow,
                img_chw.shape[2],
                img_chw.shape[3],
            )

        if (
            single_flow.shape[0] == 1
            and img_chw.shape[0] > 1
        ):
            single_flow = single_flow.expand(
                img_chw.shape[0], -1, -1, -1
            )

        warped, valid = warp_with_flow(
            img_chw, single_flow, mode=interpolation
        )

        return (
            image_chw_to_comfyui(warped),
            flow_field_to_comfyui_mask(valid),
        )


NODE_CLASS_MAPPINGS = {
    "MegaFlowWarp": MegaFlowWarp,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MegaFlowWarp": "MegaFlow Warp",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

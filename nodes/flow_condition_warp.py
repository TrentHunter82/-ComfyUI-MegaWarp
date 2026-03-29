"""
Flow Condition Warp -- Warp ControlNet conditioning maps
(depth, canny, normals) using pre-computed optical flow.
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
)


class FlowConditionWarp:
    """
    Warp ControlNet condition maps using optical flow.
    Run expensive preprocessing (DepthAnything, Canny) on
    keyframes only, then warp the results to intermediate
    frames. Produces temporally smooth conditioning with
    massive speed savings.
    """

    CATEGORY = "MegaWarp/ControlNet"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("warped_condition",)
    OUTPUT_TOOLTIPS = (
        "Flow-warped conditioning image for ControlNet",
    )
    FUNCTION = "warp_condition"
    DESCRIPTION = (
        "Warp a ControlNet condition map (depth, canny, "
        "normals) using pre-computed optical flow. "
        "Avoids re-running expensive preprocessing on "
        "every frame while maintaining temporal "
        "consistency."
    )

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "condition_image": ("IMAGE",),
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

    def warp_condition(
        self,
        condition_image: torch.Tensor,
        flow: torch.Tensor,
        frame_index: int = 0,
        interpolation: str = "bilinear",
    ) -> Tuple[torch.Tensor]:
        img_chw = comfyui_to_image_chw(condition_image)

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

        warped, _ = warp_with_flow(
            img_chw, single_flow, mode=interpolation
        )

        return (image_chw_to_comfyui(warped),)


NODE_CLASS_MAPPINGS = {
    "FlowConditionWarp": FlowConditionWarp,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlowConditionWarp": "Flow Condition Warp",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

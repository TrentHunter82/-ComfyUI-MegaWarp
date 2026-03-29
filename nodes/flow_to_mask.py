"""
Flow To Mask -- Convert optical flow magnitude to a
motion segmentation mask.
"""

import torch
from typing import Any, Dict, Tuple

from ..utils.warp_ops import flow_magnitude
from ..utils.tensor_utils import (
    select_flow_pair,
    flow_field_to_comfyui_mask,
)


class FlowToMask:
    """
    Convert the magnitude of an optical flow field into a
    soft motion mask. Pixels with displacement above the
    threshold become white; a sigmoid falloff produces
    smooth boundaries.
    """

    CATEGORY = "MegaWarp/Utils"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("motion_mask",)
    OUTPUT_TOOLTIPS = (
        "Motion mask: white where displacement exceeds "
        "threshold",
    )
    FUNCTION = "to_mask"
    DESCRIPTION = (
        "Convert optical flow magnitude to a motion "
        "mask. Use to isolate moving regions for "
        "selective stylization -- e.g. heavy style on "
        "background, keep subject natural."
    )

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "flow": ("FLOW_FIELD",),
            },
            "optional": {
                "threshold": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 200.0,
                    "step": 0.5,
                    "tooltip": (
                        "Pixels with displacement above "
                        "this value become white"
                    ),
                }),
                "softness": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.1,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": (
                        "Sigmoid falloff width. Higher "
                        "= smoother mask edges"
                    ),
                }),
                "invert": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Invert the mask so static "
                        "regions become white"
                    ),
                }),
                "frame_index": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 9999,
                    "tooltip": (
                        "-1 = all pairs, >=0 = "
                        "specific pair"
                    ),
                }),
            },
        }

    def to_mask(
        self,
        flow: torch.Tensor,
        threshold: float = 5.0,
        softness: float = 2.0,
        invert: bool = False,
        frame_index: int = -1,
    ) -> Tuple[torch.Tensor]:
        if frame_index >= 0:
            idx = min(frame_index, flow.shape[0] - 1)
            flow = select_flow_pair(flow, idx)

        mag = flow_magnitude(flow)  # [B, 1, H, W]
        mask = torch.sigmoid(
            (mag - threshold) / max(softness, 1e-5)
        )

        if invert:
            mask = 1.0 - mask

        return (flow_field_to_comfyui_mask(mask),)


NODE_CLASS_MAPPINGS = {
    "FlowToMask": FlowToMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlowToMask": "Flow To Mask",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

"""
Flow Visualize -- Render an optical flow field as a
color-coded image for debugging and preview.
"""

import torch
from typing import Any, Dict, Tuple

from ..utils.warp_ops import flow_to_hsv_image, flow_magnitude
from ..utils.tensor_utils import (
    image_chw_to_comfyui,
    select_flow_pair,
)


class FlowVisualize:
    """
    Convert a dense optical flow field into a human-readable
    color image. HSV wheel maps direction to hue and
    magnitude to brightness. Magnitude mode shows a
    grayscale intensity map.
    """

    CATEGORY = "MegaWarp/Utils"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("flow_image",)
    OUTPUT_TOOLTIPS = (
        "Color-coded visualization of the flow field",
    )
    FUNCTION = "visualize"
    DESCRIPTION = (
        "Render optical flow as a color image. HSV "
        "wheel encodes flow direction as hue and "
        "magnitude as brightness. Magnitude mode shows "
        "a grayscale displacement map. Useful for "
        "debugging flow quality."
    )

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "flow": ("FLOW_FIELD",),
            },
            "optional": {
                "frame_index": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 9999,
                    "tooltip": (
                        "-1 = visualize all pairs as a "
                        "batch, >=0 = specific pair"
                    ),
                }),
                "method": (
                    ["hsv_wheel", "magnitude"],
                    {"default": "hsv_wheel"},
                ),
                "max_flow": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 500.0,
                    "step": 1.0,
                    "tooltip": (
                        "0 = auto-normalize from max "
                        "displacement"
                    ),
                }),
            },
        }

    def visualize(
        self,
        flow: torch.Tensor,
        frame_index: int = -1,
        method: str = "hsv_wheel",
        max_flow: float = 0.0,
    ) -> Tuple[torch.Tensor]:
        if frame_index >= 0:
            idx = min(frame_index, flow.shape[0] - 1)
            vis_flow = select_flow_pair(flow, idx)
        else:
            vis_flow = flow

        mf = max_flow if max_flow > 0 else None

        if method == "hsv_wheel":
            rgb = flow_to_hsv_image(vis_flow, max_flow=mf)
        else:
            mag = flow_magnitude(vis_flow)
            if mf is None:
                mf = mag.max().item()
            mf = max(mf, 1e-5)
            gray = torch.clamp(mag / mf, 0, 1)
            rgb = gray.expand(-1, 3, -1, -1)

        return (image_chw_to_comfyui(rgb),)


NODE_CLASS_MAPPINGS = {
    "FlowVisualize": FlowVisualize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlowVisualize": "Flow Visualize",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

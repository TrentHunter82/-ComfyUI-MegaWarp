"""
MegaFlow Consistency -- Generate soft confidence masks
from flow field analysis, replacing binary forward/backward
consistency checks.
"""

import torch
from typing import Any, Dict, Tuple

from ..utils.warp_ops import (
    compute_flow_divergence,
    flow_magnitude,
)
from ..utils.tensor_utils import (
    select_flow_pair,
    flow_field_to_comfyui_mask,
)


class MegaFlowConsistency:
    """
    Produce a soft [0,1] confidence mask from an optical
    flow field. High confidence indicates reliable flow;
    low confidence marks occlusions, motion boundaries,
    and unreliable regions. Two methods available:
    divergence-based (fast, no backward pass) and
    magnitude-based thresholding.
    """

    CATEGORY = "MegaWarp/Flow"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("confidence_mask",)
    OUTPUT_TOOLTIPS = (
        "Soft confidence mask: 1 = reliable flow, "
        "0 = likely occluded or unreliable",
    )
    FUNCTION = "compute_consistency"
    DESCRIPTION = (
        "Generate a soft confidence mask from flow "
        "field analysis. Replaces WarpFusion's binary "
        "forward/backward consistency check with "
        "smooth gradients at occlusion boundaries. "
        "Feed into Flow Blend for adaptive compositing."
    )

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "flow": ("FLOW_FIELD",),
            },
            "optional": {
                "method": (
                    [
                        "flow_divergence",
                        "magnitude_threshold",
                    ],
                    {"default": "flow_divergence"},
                ),
                "threshold": ("FLOAT", {
                    "default": 3.0,
                    "min": 0.1,
                    "max": 50.0,
                    "step": 0.1,
                    "tooltip": (
                        "For divergence: scales the "
                        "gradient response. For "
                        "magnitude: displacement cutoff."
                    ),
                }),
                "softness": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": (
                        "Sigmoid falloff width. Higher "
                        "= smoother transition at "
                        "occlusion boundaries"
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

    def compute_consistency(
        self,
        flow: torch.Tensor,
        method: str = "flow_divergence",
        threshold: float = 3.0,
        softness: float = 1.0,
        frame_index: int = -1,
    ) -> Tuple[torch.Tensor]:
        if frame_index >= 0:
            idx = min(frame_index, flow.shape[0] - 1)
            flow = select_flow_pair(flow, idx)

        if method == "flow_divergence":
            confidence = compute_flow_divergence(
                flow, softness=softness
            )
        else:
            # magnitude_threshold: large flow = low
            # confidence (likely error or extreme motion)
            mag = flow_magnitude(flow)
            confidence = 1.0 - torch.sigmoid(
                (mag - threshold)
                / max(softness, 1e-5)
            )

        return (flow_field_to_comfyui_mask(confidence),)


NODE_CLASS_MAPPINGS = {
    "MegaFlowConsistency": MegaFlowConsistency,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MegaFlowConsistency": "MegaFlow Consistency",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

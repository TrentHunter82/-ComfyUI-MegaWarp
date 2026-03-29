"""
MegaFlow Estimate -- Compute dense optical flow from a
video frame sequence using MegaFlow temporal attention.
"""

import torch
from typing import Any, Dict, Tuple

from ..utils.tensor_utils import (
    comfyui_images_to_megaflow_video,
    flow_field_to_comfyui_mask,
)
from ..utils.warp_ops import compute_flow_divergence


class MegaFlowEstimate:
    """
    Run MegaFlow inference on a batch of video frames to
    produce dense optical flow fields for each consecutive
    frame pair. Supports chunked processing for long
    sequences on memory-constrained GPUs.
    """

    CATEGORY = "MegaWarp/Flow"
    RETURN_TYPES = ("FLOW_FIELD", "MASK")
    RETURN_NAMES = ("flow", "confidence")
    OUTPUT_TOOLTIPS = (
        "Dense optical flow [T-1, 2, H, W] for each "
        "consecutive frame pair",
        "Soft confidence mask derived from flow "
        "divergence -- low at occlusion boundaries",
    )
    FUNCTION = "estimate"
    DESCRIPTION = (
        "Compute dense optical flow between consecutive "
        "video frames using MegaFlow. Returns flow "
        "fields and a soft confidence mask. Use chunked "
        "processing for sequences over 30 frames on "
        "16 GB GPUs."
    )

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MEGAFLOW_MODEL",),
                "images": ("IMAGE",),
            },
            "optional": {
                "num_refine_iters": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 32,
                    "step": 1,
                    "tooltip": (
                        "More iterations = more accurate "
                        "flow, slower. 4-6 for preview, "
                        "8+ for production."
                    ),
                }),
                "use_bf16": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Use bfloat16 for faster "
                        "inference. Disable for maximum "
                        "precision."
                    ),
                }),
                "chunk_size": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 128,
                    "step": 1,
                    "tooltip": (
                        "0 = process all frames at once. "
                        ">0 = chunk with overlap to save "
                        "VRAM."
                    ),
                }),
                "chunk_overlap": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 16,
                    "step": 1,
                }),
            },
        }

    def estimate(
        self,
        model: Any,
        images: torch.Tensor,
        num_refine_iters: int = 8,
        use_bf16: bool = True,
        chunk_size: int = 0,
        chunk_overlap: int = 4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(model.parameters()).device

        video = comfyui_images_to_megaflow_video(
            images
        ).to(device)

        num_frames = video.shape[1]
        if num_frames < 2:
            raise ValueError(
                "MegaFlowEstimate requires at least "
                f"2 frames, got {num_frames}"
            )

        with torch.inference_mode():
            with torch.autocast(
                device_type=str(device),
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                if (
                    chunk_size <= 0
                    or num_frames <= chunk_size
                ):
                    result = model(
                        video,
                        num_reg_refine=num_refine_iters,
                    )
                    # [1, T-1, 2, H, W] -> [T-1, 2, H, W]
                    flow = result["flow_preds"][
                        -1
                    ].squeeze(0)
                else:
                    flow = self._chunked_estimate(
                        model,
                        video,
                        chunk_size,
                        chunk_overlap,
                        num_refine_iters,
                    )

        confidence = compute_flow_divergence(
            flow.float()
        )
        confidence_mask = flow_field_to_comfyui_mask(
            confidence
        )

        return (
            flow.float().cpu(),
            confidence_mask.float().cpu(),
        )

    def _chunked_estimate(
        self,
        model: Any,
        video: torch.Tensor,
        chunk_size: int,
        overlap: int,
        num_refine_iters: int,
    ) -> torch.Tensor:
        """Process long video in overlapping chunks."""
        num_frames = video.shape[1]
        all_flows = []

        start = 0
        while start < num_frames - 1:
            end = min(start + chunk_size, num_frames)
            chunk = video[:, start:end]

            result = model(
                chunk,
                num_reg_refine=num_refine_iters,
            )
            chunk_flow = result["flow_preds"][
                -1
            ].squeeze(0)

            if start == 0:
                all_flows.append(chunk_flow)
            else:
                skip = min(overlap, chunk_flow.shape[0])
                if skip < chunk_flow.shape[0]:
                    all_flows.append(chunk_flow[skip:])

            start += chunk_size - overlap
            if end >= num_frames:
                break

        return torch.cat(all_flows, dim=0)


NODE_CLASS_MAPPINGS = {
    "MegaFlowEstimate": MegaFlowEstimate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MegaFlowEstimate": "MegaFlow Estimate",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

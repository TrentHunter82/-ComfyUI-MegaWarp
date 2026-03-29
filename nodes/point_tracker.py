"""
MegaFlow Track -- Dense point tracking across video frames
using MegaFlow's temporal attention architecture.
"""

import torch
from typing import Any, Dict, Tuple

from ..utils.tensor_utils import (
    comfyui_images_to_megaflow_video,
    flow_field_to_comfyui_mask,
)


class MegaFlowTrack:
    """
    Track all pixels across a video sequence using
    MegaFlow's point tracking mode. Generates dense
    trajectories and per-pixel motion masks for
    selective stylization, subject stabilization, or
    ControlNet pose estimation.
    """

    CATEGORY = "MegaWarp/Track"
    RETURN_TYPES = ("TRACKS", "MASK")
    RETURN_NAMES = ("tracks", "motion_mask")
    OUTPUT_TOOLTIPS = (
        "Dense point trajectories [1, T, 2, H, W]",
        "Motion mask: white where displacement "
        "exceeds threshold",
    )
    FUNCTION = "track"
    DESCRIPTION = (
        "Track all pixels across video frames using "
        "MegaFlow point tracking. Outputs dense "
        "trajectories and a motion mask. Requires the "
        "megaflow-track model variant."
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
                        "More iterations = more "
                        "accurate tracking, slower."
                    ),
                }),
                "motion_threshold": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.5,
                    "tooltip": (
                        "Pixels with total "
                        "displacement above this "
                        "become white in the mask"
                    ),
                }),
                "use_bf16": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Use bfloat16 for faster "
                        "inference"
                    ),
                }),
            },
        }

    def track(
        self,
        model: Any,
        images: torch.Tensor,
        num_refine_iters: int = 8,
        motion_threshold: float = 5.0,
        use_bf16: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            from megaflow.utils.basic import gridcloud2d
        except ImportError:
            raise ImportError(
                "MegaFlow tracking requires "
                "megaflow.utils.basic.gridcloud2d. "
                "Ensure MegaFlow is installed: "
                "pip install "
                "git+https://github.com/cvg/"
                "megaflow.git"
            )

        device = next(model.parameters()).device
        video = comfyui_images_to_megaflow_video(
            images
        ).to(device)

        num_frames = video.shape[1]
        if num_frames < 2:
            raise ValueError(
                "MegaFlowTrack requires at least "
                f"2 frames, got {num_frames}"
            )

        _, _, _, img_h, img_w = video.shape

        with torch.inference_mode():
            with torch.autocast(
                device_type=str(device),
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                result = model.forward_track(
                    video,
                    num_reg_refine=num_refine_iters,
                )

        flows_e = result["flow_final"]

        # Build identity grid and compute trajectories
        grid_xy = gridcloud2d(
            1, img_h, img_w,
            norm=False, device=device,
        ).float()
        grid_xy = grid_xy.permute(0, 2, 1).reshape(
            1, 1, 2, img_h, img_w
        )
        tracks = flows_e + grid_xy  # [1, T, 2, H, W]

        # Motion mask: max displacement from start
        displacement = tracks - grid_xy  # [1, T, 2, H, W]
        disp_mag = torch.norm(
            displacement, dim=2
        )  # [1, T, H, W]
        max_disp = disp_mag.max(dim=1).values  # [1, H, W]

        motion_mask = torch.sigmoid(
            (max_disp - motion_threshold) / 2.0
        )

        return (
            tracks.float().cpu(),
            motion_mask.float().cpu(),
        )


NODE_CLASS_MAPPINGS = {
    "MegaFlowTrack": MegaFlowTrack,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MegaFlowTrack": "MegaFlow Track",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

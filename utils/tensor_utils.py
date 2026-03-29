"""
MegaWarp — Tensor format conversion between ComfyUI and MegaFlow.

ComfyUI conventions:
  IMAGE:  [B, H, W, C] float32 [0, 1]  RGB channels-last
  LATENT: {"samples": [B, 4, H/8, W/8]}
  MASK:   [B, H, W] or [H, W] float32 [0, 1]

MegaFlow conventions:
  Video:  [1, T, 3, H, W] float32 [0, 255]  RGB channels-first
  Flow:   [T-1, 2, H, W]  (dx, dy) pixel displacements
"""

import torch


def comfyui_images_to_megaflow_video(images: torch.Tensor) -> torch.Tensor:
    """
    Convert ComfyUI image batch to MegaFlow video tensor.
    
    Args:
        images: [B, H, W, C] float32 [0, 1]
    
    Returns:
        video: [1, T, 3, H, W] float32 [0, 255]
    """
    # [B, H, W, 3] → [B, 3, H, W]
    video = images.permute(0, 3, 1, 2).contiguous()
    # Scale to [0, 255] and add batch dim
    video = video.unsqueeze(0) * 255.0  # [1, T, 3, H, W]
    return video


def megaflow_flow_to_flow_field(flow_preds: torch.Tensor) -> torch.Tensor:
    """
    Extract flow field from MegaFlow output.
    
    MegaFlow returns flow_preds as a list of refinement iterations.
    We take the last one (most refined).
    
    Args:
        flow_preds: last element of result["flow_preds"] — [T-1, 2, H, W]
    
    Returns:
        flow: [T-1, 2, H, W] — our FLOW_FIELD custom type
    """
    # flow_preds[-1] is already [T-1, 2, H, W]
    # Channel 0 = horizontal displacement (dx)
    # Channel 1 = vertical displacement (dy)
    return flow_preds


def flow_field_to_comfyui_mask(confidence: torch.Tensor) -> torch.Tensor:
    """
    Convert a [B, 1, H, W] confidence map to ComfyUI MASK format [B, H, W].
    """
    if confidence.dim() == 4 and confidence.shape[1] == 1:
        return confidence.squeeze(1)
    return confidence


def comfyui_mask_to_4d(mask: torch.Tensor) -> torch.Tensor:
    """
    Convert ComfyUI MASK [B, H, W] or [H, W] to [B, 1, H, W] for operations.
    """
    if mask.dim() == 2:
        return mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        return mask.unsqueeze(1)
    return mask


def image_chw_to_comfyui(img: torch.Tensor) -> torch.Tensor:
    """
    Convert [B, C, H, W] float [0, 1] to ComfyUI [B, H, W, C].
    """
    return img.permute(0, 2, 3, 1).contiguous()


def comfyui_to_image_chw(img: torch.Tensor) -> torch.Tensor:
    """
    Convert ComfyUI [B, H, W, C] to [B, C, H, W].
    """
    return img.permute(0, 3, 1, 2).contiguous()


def select_flow_pair(flow: torch.Tensor, frame_index: int) -> torch.Tensor:
    """
    Select a single flow field from a batch.
    
    Flow from MegaFlowEstimate is [T-1, 2, H, W] for all consecutive pairs.
    This selects flow for pair (frame_index → frame_index+1).
    
    Args:
        flow: [T-1, 2, H, W]
        frame_index: which pair (0 = frames 0→1, 1 = frames 1→2, etc.)
    
    Returns:
        single_flow: [1, 2, H, W]
    """
    return flow[frame_index : frame_index + 1]


def ensure_device_match(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Move all tensors to the device of the first tensor."""
    target_device = tensors[0].device
    return tuple(t.to(target_device) if t.device != target_device else t for t in tensors)

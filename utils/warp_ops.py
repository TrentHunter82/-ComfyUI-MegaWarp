"""
MegaWarp — Core warping operations.
Shared by MegaFlowWarp, MegaFlowLatentWarp, FlowNoiseWarp, FlowConditionWarp.
"""

import torch
import torch.nn.functional as F


def warp_with_flow(x: torch.Tensor, flow: torch.Tensor, mode: str = "bilinear") -> tuple[torch.Tensor, torch.Tensor]:
    """
    Warp tensor x according to dense optical flow field.
    
    Args:
        x: [B, C, H, W] source tensor to warp
        flow: [B, 2, H, W] displacement field in pixels (channel 0 = dx, channel 1 = dy)
        mode: interpolation mode — "bilinear" or "bicubic"
    
    Returns:
        warped: [B, C, H, W] warped result
        valid_mask: [B, 1, H, W] binary mask where 1 = valid (source pixel existed)
    """
    B, C, H, W = x.shape
    
    # Build identity grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=x.device, dtype=x.dtype),
        torch.arange(W, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    # [1, 2, H, W] — (x, y) order to match flow convention
    base_grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
    
    # Displaced grid
    vgrid = base_grid + flow  # [B, 2, H, W]
    
    # Normalize to [-1, 1] for F.grid_sample
    vgrid_norm = vgrid.clone()
    vgrid_norm[:, 0, :, :] = 2.0 * vgrid_norm[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid_norm[:, 1, :, :] = 2.0 * vgrid_norm[:, 1, :, :] / max(H - 1, 1) - 1.0
    vgrid_norm = vgrid_norm.permute(0, 2, 3, 1)  # [B, H, W, 2]
    
    warped = F.grid_sample(x, vgrid_norm, mode=mode, padding_mode="zeros", align_corners=True)
    
    # Valid mask: warp a tensor of ones to find which pixels had valid sources
    ones = torch.ones(B, 1, H, W, device=x.device, dtype=x.dtype)
    valid_mask = F.grid_sample(ones, vgrid_norm, mode="nearest", padding_mode="zeros", align_corners=True)
    
    return warped, valid_mask


def scale_flow_to_resolution(flow: torch.Tensor, target_h: int, target_w: int, mode: str = "area") -> torch.Tensor:
    """
    Rescale a flow field to a different spatial resolution (e.g., for latent-space warping).
    
    Crucially, this scales both the spatial dimensions AND the displacement magnitudes.
    
    Args:
        flow: [B, 2, H, W] flow field at original resolution
        target_h: target height
        target_w: target width
        mode: interpolation mode for spatial rescaling
    
    Returns:
        scaled_flow: [B, 2, target_h, target_w] with proportionally scaled displacements
    """
    _, _, H, W = flow.shape
    
    # Spatially resize
    scaled_flow = F.interpolate(flow, size=(target_h, target_w), mode=mode)
    
    # Scale displacement magnitudes proportionally
    scale_x = target_w / W
    scale_y = target_h / H
    scaled_flow[:, 0, :, :] *= scale_x
    scaled_flow[:, 1, :, :] *= scale_y
    
    return scaled_flow


def compute_flow_divergence(flow: torch.Tensor, softness: float = 1.0) -> torch.Tensor:
    """
    Compute a soft confidence mask from flow field divergence.
    
    High divergence indicates occlusion boundaries, disoccluded regions, or
    unreliable flow. This replaces WarpFusion's binary forward/backward consistency check.
    
    Args:
        flow: [B, 2, H, W] flow field
        softness: controls the sigmoid falloff width
    
    Returns:
        confidence: [B, 1, H, W] in [0, 1] where 1 = confident, 0 = likely occluded
    """
    # Spatial gradients of flow (Sobel-like)
    du_dx = flow[:, 0:1, :, 1:] - flow[:, 0:1, :, :-1]  # d(flow_x)/dx
    dv_dy = flow[:, 1:2, 1:, :] - flow[:, 1:2, :-1, :]  # d(flow_y)/dy
    
    # Pad to original size
    du_dx = F.pad(du_dx, (0, 1, 0, 0), mode="replicate")
    dv_dy = F.pad(dv_dy, (0, 0, 0, 1), mode="replicate")
    
    # Divergence magnitude
    divergence = (du_dx.abs() + dv_dy.abs())
    
    # Also consider cross-terms for shear
    du_dy = flow[:, 0:1, 1:, :] - flow[:, 0:1, :-1, :]
    dv_dx = flow[:, 1:2, :, 1:] - flow[:, 1:2, :, :-1]
    du_dy = F.pad(du_dy, (0, 0, 0, 1), mode="replicate")
    dv_dx = F.pad(dv_dx, (0, 1, 0, 0), mode="replicate")
    
    total_gradient = divergence + 0.5 * (du_dy.abs() + dv_dx.abs())
    
    # Soft confidence via sigmoid
    confidence = torch.sigmoid(-total_gradient / softness + 2.0)
    
    return confidence


def flow_magnitude(flow: torch.Tensor) -> torch.Tensor:
    """
    Compute per-pixel flow magnitude.
    
    Args:
        flow: [B, 2, H, W]
    
    Returns:
        magnitude: [B, 1, H, W]
    """
    return torch.norm(flow, dim=1, keepdim=True)


def flow_to_hsv_image(flow: torch.Tensor, max_flow: float | None = None) -> torch.Tensor:
    """
    Convert flow field to HSV color wheel visualization.
    
    Args:
        flow: [B, 2, H, W]
        max_flow: normalization factor. None = auto from max magnitude.
    
    Returns:
        rgb: [B, 3, H, W] float [0, 1]
    """
    B, _, H, W = flow.shape
    
    fx = flow[:, 0]  # [B, H, W]
    fy = flow[:, 1]
    
    mag = torch.sqrt(fx ** 2 + fy ** 2)
    if max_flow is None:
        max_flow = mag.max().item()
    max_flow = max(max_flow, 1e-5)
    
    # Angle → hue [0, 1]
    angle = torch.atan2(fy, fx)  # [-pi, pi]
    hue = (angle / (2 * 3.14159265) + 0.5) % 1.0  # [0, 1]
    
    # Magnitude → value [0, 1]
    value = torch.clamp(mag / max_flow, 0, 1)
    
    # Saturation = 1
    saturation = torch.ones_like(hue)
    
    # HSV → RGB conversion
    hsv = torch.stack([hue, saturation, value], dim=1)  # [B, 3, H, W]
    
    h = hsv[:, 0] * 6.0
    s = hsv[:, 1]
    v = hsv[:, 2]
    
    hi = torch.floor(h).long() % 6
    f = h - torch.floor(h)
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    
    rgb = torch.zeros(B, 3, H, W, device=flow.device, dtype=flow.dtype)
    
    for i in range(6):
        mask = (hi == i).unsqueeze(1).float()
        if i == 0:
            rgb += mask * torch.stack([v, t, p], dim=1)
        elif i == 1:
            rgb += mask * torch.stack([q, v, p], dim=1)
        elif i == 2:
            rgb += mask * torch.stack([p, v, t], dim=1)
        elif i == 3:
            rgb += mask * torch.stack([p, q, v], dim=1)
        elif i == 4:
            rgb += mask * torch.stack([t, p, v], dim=1)
        elif i == 5:
            rgb += mask * torch.stack([v, p, q], dim=1)
    
    return rgb

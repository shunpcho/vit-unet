from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class VitunetConfig:
    """Configuration for ViTUNet model."""

    depth: int
    depth_te: int
    size_bottleneck: int
    preprocessing: Literal["conv", "fourier", "none"]
    im_size: int
    patch_size: int
    num_channels: int
    hidden_dim: int
    num_heads: int
    attn_drop: float
    proj_drop: float
    linear_drop: float
    verbose: bool = False

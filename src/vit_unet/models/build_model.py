from typing import Literal

from vit_unet.config.model_config import VitunetConfig
from vit_unet.models.model import ViTUNet


def get_vit_unet(
    model_string: Literal["lite", "base", "large"], verbose: bool = False, im_size: int | None = None
) -> ViTUNet:
    if model_string == "lite":
        config = VitunetConfig(
            depth=2,
            depth_te=1,
            size_bottleneck=2,
            preprocessing="conv",
            im_size=im_size if im_size is not None else 256,
            patch_size=16,
            num_channels=3,
            hidden_dim=64,
            num_heads=4,
            attn_drop=0.2,
            proj_drop=0.2,
            linear_drop=0,
            verbose=verbose,
        )
        return ViTUNet(config)

    if model_string == "base":
        config = VitunetConfig(
            depth=2,
            depth_te=2,
            size_bottleneck=2,
            preprocessing="conv",
            im_size=im_size if im_size is not None else 256,
            patch_size=32,
            num_channels=3,
            hidden_dim=128,
            num_heads=8,
            attn_drop=0.2,
            proj_drop=0.2,
            linear_drop=0,
            verbose=verbose,
        )
        return ViTUNet(config)

    if model_string == "large":
        config = VitunetConfig(
            depth=2,
            depth_te=4,
            size_bottleneck=4,
            preprocessing="conv",
            im_size=im_size if im_size is not None else 256,
            patch_size=32,
            num_channels=3,
            hidden_dim=256,
            num_heads=8,
            attn_drop=0.2,
            proj_drop=0.2,
            linear_drop=0,
            verbose=verbose,
        )
        return ViTUNet(config)

    msg = f"Model string {model_string} not valid"
    raise ValueError(msg)

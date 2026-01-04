import torch

from vit_unet.config.model_config import VitunetConfig
from vit_unet.models.model import ViTUNet


def test_vit_unet_model():
    # Define configuration
    config = VitunetConfig(
        depth=2,
        depth_te=2,
        size_bottleneck=4,
        im_size=224,
        patch_size=32,
        num_channels=3,
        hidden_dim=3072,
        num_heads=12,
        linear_drop=0.1,
        attn_drop=0.2,
        proj_drop=0.2,
        preprocessing="conv",
        verbose=True,
    )

    # Initialize model
    model = ViTUNet(config)

    # Test forward pass
    x = torch.randn(2, 3, 224, 224)

    # Test Patch Encoder
    embedded_patches = model.patch_encoder(x)
    print(f"Embedded patches shape: {embedded_patches.shape}")

    # Test Encoder Block
    # Level 1
    x = model._encoder_patches(embedded_patches)

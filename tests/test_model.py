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
    print("Testing Encoder Blocks")
    x_patch, encoder_skip = model._encode_patches(embedded_patches)
    print(f"Encoder output x_patch shape: {x_patch.shape}")
    for i, skip in enumerate(encoder_skip):
        print(f"Skip connection {i + 1} shape: {skip.shape}")

    # Test Bottleneck
    print("Testing Bottleneck")
    x_bottleneck = model._process_bottleneck(x_patch)
    print(f"Bottleneck output x_bottleneck shape: {x_bottleneck.shape}")

    # Test Decoder Block
    print("Testing Decoder Blocks")
    x_decoded = model._decode_patches(x_bottleneck, encoder_skip)
    print(f"Decoder output x_decoded shape: {x_decoded.shape}")


if __name__ == "__main__":
    test_vit_unet_model()

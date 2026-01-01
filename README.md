# ViT-UNet

ViT-UNet is a model for ViT-based image restoration tasks applied to autoencoders via a UNet-type architecture.

Source: [vit-unet](https://github.com/benayas1/vit-unet)

## Overview

This module implements a Vision Transformer (ViT) based U-Net architecture for image denoising tasks. The model combines the benefits of transformer attention mechanisms with the U-Net encoder-decoder structure and skip connections.

## Architecture Components

### Core Building Blocks

#### 1. **Patch Operations**

- `patch(x, patch_size)` - Splits images into patches for transformer processing
- `unpatch(x, num_channels)` - Reconstructs images from patches
- `unflatten(flattened, num_channels)` - Reshapes flattened patches back to 5D tensors
- `downsampling(patches, num_channels)` - Reduces spatial resolution
- `upsampling(patches, num_channels)` - Increases spatial resolution

#### 2. **PatchEncoder**

Initial encoding layer that:

- Applies optional preprocessing (conv/fourier/none)
- Divides input images into patches
- Adds positional embeddings
- Returns encoded patch representations

**Parameters:**

- `img_size` - Input image dimension
- `patch_size` - Size of each patch (e.g., 16x16, 32x32)
- `num_channels` - Number of color channels (typically 3 for RGB)
- `preprocessing` - Preprocessing method: `"conv"`, `"fourier"`, or `"none"`

#### 3. **ReAttention**

Custom multi-head attention mechanism with:

- Spatial convolutions on Q, K, V inputs
- Optional re-attention matrix transformation
- Batch normalization for stability
- Dropout for regularization

**Key Features:**

- Processes patches through 2D convolutions before attention
- Supports re-attention scaling for enhanced feature learning
- Returns both output and attention weights

#### 4. **ReAttentionTransformerEncoder**

Combines ReAttention with FeedForward network:

- Multi-head attention with residual connections
- Layer normalization for training stability
- FeedForward network with GELU activation
- Double residual connections (attention + FFN)

#### 5. **SkipConnection**

U-Net style skip connections using attention:

- Connects encoder and decoder at matching resolutions
- Applies attention between encoder output and decoder input
- Maintains spatial information across network depth

#### 6. **FeedForward**

Simple two-layer MLP with:

- Linear projection to hidden dimension
- GELU activation
- Dropout regularization
- Linear projection back to original dimension

### Main Model: ViTUNet

The complete architecture organized into clean, modular methods:

#### Initialization (`__init__`)

Orchestrates model construction through helper methods:

1. `_validate_config()` - Validates configuration parameters
2. `_setup_dimensions()` - Calculates derived dimensions
3. `_print_architecture_info()` - Displays architecture details
4. `_build_layers()` - Constructs all model layers

#### Configuration Validation (`_validate_config`)

Checks:

- Patch size divisibility by depth factor
- Final patch size meets minimum requirement (≥4)
- Image size divisible by patch size
- Valid preprocessing method

#### Layer Construction Methods

**`_build_patch_encoder()`**

- Creates initial PatchEncoder
- Configures preprocessing method

**`_build_encoder()`**

- Constructs encoder transformer blocks
- Applies exponential scaling for multi-resolution processing
- Each level: `depth_te` transformer encoder blocks

**`_build_bottleneck()`**

- Creates bottleneck transformer blocks
- Operates at coarsest resolution
- Number of blocks: `size_bottleneck`

**`_build_decoder()`**

- Constructs decoder transformer blocks
- Creates skip connection modules
- Mirrors encoder structure in reverse

**`_build_output()`**

- Optional final convolution layer
- Only added if preprocessing == "conv"

#### Forward Pass Methods

**`forward(x)`**
Main forward pass:

1. Resize input to target size
2. Apply patch encoder
3. Encode with downsampling
4. Process through bottleneck
5. Decode with upsampling and skip connections
6. Reconstruct final image

**`_encode_patches(x_patch)`**

- Applies encoder blocks sequentially
- Collects intermediate outputs for skip connections
- Downsamples after each encoder level

**`_process_bottleneck(x_patch)`**

- Applies bottleneck transformer blocks
- Operates at lowest resolution

**`_decode_patches(x_patch, encoder_skip)`**

- Applies decoder blocks sequentially
- Integrates skip connections from encoder
- Upsamples after each decoder level

**`_apply_skip_connection(x_patch, encoder_skip, i)`**

- Upsamples decoder features
- Validates shape compatibility
- Applies skip connection attention

**`_apply_final_processing(x_patch, batch, x)`**

- Reconstructs image from patches
- Applies final preprocessing (if enabled)
- Returns denoised output

## Model Configuration

### Typical Parameters

```python
config = VitunetConfig(
    depth=2,              # Number of encoder/decoder levels
    depth_te=2,           # Transformer blocks per level
    size_bottleneck=4,    # Bottleneck transformer blocks
    im_size=224,          # Input image size (224 or 256)
    patch_size=16,        # Patch size (16 or 32)
    num_channels=3,       # RGB channels
    projection_dim=768,   # Embedding dimension
    hidden_dim=3072,      # FFN hidden dimension
    num_heads=12,         # Attention heads
    attn_drop=0.0,        # Attention dropout
    proj_drop=0.0,        # Projection dropout
    linear_drop=0.1,      # FFN dropout
    preprocessing="conv", # Preprocessing method
    verbose=False         # Print debug info
)
```

### Model Variants

**Lite** (224x224, patch 16)

- Faster training
- Lower memory usage
- Suitable for smaller datasets

**Base/Large** (256x256, patch 32)

- Higher capacity
- Better performance on complex tasks
- Requires more GPU memory

## Architecture Scaling

The model uses exponential scaling factors:

**Encoder (level 0 → depth)**

- Patches: `num_patches × 4^level`
- Projection dim: `projection_dim / 4^level`
- Hidden dim: `hidden_dim / 2^level`
- Example: Level 0: 196 patches, Level 1: 784 patches

**Bottleneck (level = depth)**

- Maximum patches
- Minimum projection dimension
- Processes coarsest features

**Decoder (level depth → 0)**

- Mirror of encoder
- Gradual upsampling
- Skip connections from corresponding encoder level

## Code Quality Features

### Refactored Design

- **Single Responsibility**: Each method has one clear purpose
- **Low Complexity**: `__init__` reduced from 12+ branches to 4 simple calls
- **Testability**: Individual components can be tested independently
- **Readability**: Clear method names describe intent
- **Maintainability**: Easy to modify specific components

### Helper Method Organization

- **Validation**: `_validate_config()`
- **Setup**: `_setup_dimensions()`, `_print_architecture_info()`
- **Construction**: `_build_*()` methods
- **Forward Pass**: `_encode_*()`, `_process_*()`, `_decode_*()` methods
- **Logging**: `_log_*()` methods

### Type Safety

- Full type hints throughout
- Literal types for string enums
- TYPE_CHECKING imports for better IDE support

## Usage Example

```python
from vit_unet.models.model import ViTUNet
from vit_unet.config.model_config import VitunetConfig

# Create configuration
config = VitunetConfig(
    depth=2,
    depth_te=2,
    size_bottleneck=4,
    im_size=224,
    patch_size=16,
    num_channels=3,
    hidden_dim=768,
    num_heads=8,
    linear_drop=0.1,
    preprocessing="conv"
)

# Initialize model
model = ViTUNet(config)

# Forward pass
import torch
x = torch.randn(4, 3, 224, 224)  # Batch of 4 RGB images
output = model(x)  # Denoised images

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
# Output: torch.Size([4, 3, 224, 224])
```

## Training Considerations

### Memory Optimization

- Use gradient checkpointing for large models
- Reduce batch size if OOM
- Consider mixed precision training (fp16)

### Learning Rate

- Recommended: `1e-6` (stable training)
- Use cosine annealing schedule
- Lower rates prevent training instability

### Regularization

- Dropout: `linear_drop=0.1`
- Weight decay: `0.01`
- Gradient clipping: max norm `1.0`

### Known Issues

- High learning rates (>1e-4) cause instability
- LayerNorm requires dynamic shape support
- FFT preprocessing removed due to instability

## Performance Metrics

Typical training behavior:

- **Loss**: MSE between predicted and clean images
- **PSNR**: Peak Signal-to-Noise Ratio (higher is better)
- **Validation**: Calculated per image using `skimage.metrics.peak_signal_noise_ratio`

## References

- Original [ViT](https://arxiv.org/pdf/2010.11929): "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
- [Deep-ViT](https://arxiv.org/pdf/2103.11886): "DeepViT: Towards Deeper Vision Transformer"
- [U-Net](https://arxiv.org/pdf/1505.04597): "U-Net: Convolutional Networks for Biomedical Image Segmentation" (Ronneberger et al., 2015)
- [EViT](https://arxiv.org/pdf/2410.15036): "EVIT-UNET: U-NET LIKE EFFICIENT VISION TRANSFORMER FORM EDICAL IMAGE SEGMENTATION ON MOBILE AND EDGE DEVICES"
- [FNet](https://arxiv.org/pdf/2105.03824): FNet: Mixing Tokens with Fourier Transforms
- [Attention UNet](https://arxiv.org/pdf/1804.03999): Attention U-Net: Learning Where to Look for the Pancreas
- [Uformer](https://arxiv.org/pdf/2106.03106): Uformer: A General U-Shaped Transformer for Image Restoration

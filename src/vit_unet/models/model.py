import math

import torch
import torch.nn.functional as F
from torch import nn

from vit_unet.config.model_config import VitunetConfig

# ============================================================================
# Patch Operations
# ============================================================================


def patch(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Split images into patches."""
    batch, channels, height, width = x.shape
    x = x.reshape(batch, channels, height // patch_size, patch_size, width // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, pH, pW]
    num_patches = (height // patch_size) * (width // patch_size)
    x = x.reshape(batch, num_patches, channels * patch_size * patch_size)
    return x


def unpatch(x: torch.Tensor, num_channels: int) -> torch.Tensor:
    """Reconstruct images from patches."""
    batch, num_patches, patch_dim = x.shape
    patch_size = int(math.sqrt(patch_dim // num_channels))
    height = width = int(math.sqrt(num_patches)) * patch_size

    x = x.reshape(batch, int(math.sqrt(num_patches)), int(math.sqrt(num_patches)), num_channels, patch_size, patch_size)
    x = x.permute(0, 3, 1, 4, 2, 5)
    x = x.reshape(batch, num_channels, height, width)
    return x


def unflatten(flattened: torch.Tensor, num_channels: int) -> torch.Tensor:
    """Reshape flattened patches to 5D tensor."""
    batch, num_patches, dim = flattened.shape
    patch_size = int(math.sqrt(dim // num_channels))
    h = w = int(math.sqrt(num_patches))

    x = flattened.reshape(batch, h, w, num_channels, patch_size, patch_size)
    return x


def downsampling(patches: torch.Tensor, num_channels: int) -> torch.Tensor:
    """Reduce spatial resolution by combining patches."""
    x = unflatten(patches, num_channels)
    batch, h, w, c, ph, pw = x.shape

    # Combine 2x2 patch groups
    x = x.reshape(batch, h, 2, w, 2, c, ph // 2, pw // 2)
    x = x.permute(0, 1, 3, 5, 2, 6, 4, 7)
    x = x.reshape(batch, 2 * h * 2 * w, c * (ph // 2) * (pw // 2))
    return x


def upsampling(patches: torch.Tensor, num_channels: int) -> torch.Tensor:
    """Increase spatial resolution by splitting patches."""
    batch, num_patches, dim = patches.shape
    patch_size = int(math.sqrt(dim // num_channels)) // 2
    h = w = int(math.sqrt(num_patches))

    # Split into 2x2 groups
    x = patches.reshape(batch, h, w, num_channels, 2, patch_size, 2, patch_size)
    x = x.permute(0, 1, 4, 2, 6, 3, 5, 7)
    x = x.reshape(batch, h * 2, w * 2, num_channels, patch_size, patch_size)
    x = x.reshape(batch, (h * 2) * (w * 2), num_channels * patch_size * patch_size)
    return x


# ============================================================================
# Model Components
# ============================================================================


class PatchEncoder(nn.Module):
    """Initial encoding layer with preprocessing and positional embeddings."""

    def __init__(
        self, img_size: int, patch_size: int, num_channels: int, projection_dim: int, preprocessing: str = "none"
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.projection_dim = projection_dim
        self.preprocessing = preprocessing

        num_patches = (img_size // patch_size) ** 2
        patch_dim = num_channels * patch_size * patch_size

        # Preprocessing layers
        if preprocessing == "conv":
            self.conv = nn.Conv2d(num_channels, num_channels, 3, padding=1)

        # Linear projection
        self.projection = nn.Linear(patch_dim, projection_dim)

        # Positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, projection_dim))
        # another option:
        # self.register_buffer("positions", torch.arange(start=0, end=num_patches, step=1))
        # self.pos_embedding = torch.nn.Embedding(num_embeddings=num_patches, embedding_dim=projection_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply preprocessing
        if self.preprocessing == "conv":
            x = self.conv(x)

        # Create patches
        x = patch(x, self.patch_size)

        # Project and add positional embeddings
        x = self.projection(x)
        x = x + self.pos_embedding

        return x


class FeedForward(nn.Module):
    """Simple MLP with GELU activation."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReAttention(nn.Module):
    """Custom multi-head attention with spatial convolutions."""

    def __init__(
        self, dim: int, num_heads: int = 8, attn_drop: float = 0.0, proj_drop: float = 0.0, apply_transform: bool = True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.apply_transform = apply_transform

        # Linear or flattened QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        # if flatten:
        #     qkvconv = nn.Conv2d(dim, dim * 3, 1, padding="same") (wip)
        #     self.qkv = torch.flatten(torch.stack([qkvconv(y) for y in unflatten(x, dim)], dim=0), -3, -1).reshape()

        # Spatial convolutions for Q, K, V
        self.conv_q = nn.Conv2d(self.head_dim, self.head_dim, 3, padding=1, groups=self.head_dim)
        self.conv_k = nn.Conv2d(self.head_dim, self.head_dim, 3, padding=1, groups=self.head_dim)
        self.conv_v = nn.Conv2d(self.head_dim, self.head_dim, 3, padding=1, groups=self.head_dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if apply_transform:
            self.transform = nn.Linear(num_heads, num_heads)
            self.norm = nn.BatchNorm1d(num_heads)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, num_patches, dim = x.shape
        h = w = int(math.sqrt(num_patches))

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(batch, num_patches, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply spatial convolutions
        q_2d = q.permute(0, 1, 3, 2).reshape(batch * self.num_heads, self.head_dim, h, w)
        k_2d = k.permute(0, 1, 3, 2).reshape(batch * self.num_heads, self.head_dim, h, w)
        v_2d = v.permute(0, 1, 3, 2).reshape(batch * self.num_heads, self.head_dim, h, w)

        q = self.conv_q(q_2d).reshape(batch, self.num_heads, self.head_dim, num_patches).permute(0, 1, 3, 2)
        k = self.conv_k(k_2d).reshape(batch, self.num_heads, self.head_dim, num_patches).permute(0, 1, 3, 2)
        v = self.conv_v(v_2d).reshape(batch, self.num_heads, self.head_dim, num_patches).permute(0, 1, 3, 2)

        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply re-attention transform
        if self.apply_transform:
            attn = attn.softmax(dim=-1)
            attn = self.transform(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            attn = self.norm(attn.reshape(batch * num_patches, self.num_heads, num_patches))
            attn = attn.reshape(batch, num_patches, self.num_heads, num_patches).permute(0, 2, 1, 3)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(batch, num_patches, dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


class ReAttentionTransformerEncoder(nn.Module):
    """Transformer encoder block with ReAttention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        hidden_dim: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        linear_drop: float = 0.1,
    ):
        super().__init__()
        # num_channels * patch_size ** 2
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ReAttention(dim, num_heads, attn_drop, proj_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, hidden_dim, linear_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention block
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm)
        x = x + attn_out

        # FFN block
        x = x + self.ffn(self.norm2(x))

        return x


class SkipConnection(nn.Module):
    """U-Net style skip connection with attention."""

    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0):
        super().__init__()
        self.attn = ReAttention(dim, num_heads, attn_drop, apply_transform=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, decoder_x: torch.Tensor, encoder_x: torch.Tensor) -> torch.Tensor:
        x = decoder_x + encoder_x
        x_norm = self.norm(x)
        attn_out, _ = self.attn(x_norm)
        return decoder_x + attn_out


# ============================================================================
# Main Model
# ============================================================================


class ViTUNet(nn.Module):
    """Vision Transformer U-Net for image denoising."""

    def __init__(self, config: VitunetConfig):
        super().__init__()
        self.config = config

        # Validation and setup
        self._validate_config()
        self._setup_dimensions()
        if config.verbose:
            self._print_architecture_info()

        # Build model layers
        self._build_layers()

    def _validate_config(self):
        """Validate configuration parameters."""
        c = self.config

        # Check patch size divisibility
        final_patch_size = c.patch_size * (2**c.depth)
        if final_patch_size < 4:
            raise ValueError(f"Final patch size {final_patch_size} < 4")

        # Check image size divisibility
        if c.im_size % c.patch_size != 0:
            raise ValueError(f"Image size {c.im_size} not divisible by patch size {c.patch_size}")

        # Check preprocessing method
        if c.preprocessing not in ["conv", "fourier", "none"]:
            raise ValueError(f"Invalid preprocessing: {c.preprocessing}")

    def _setup_dimensions(self):
        """Calculate derived dimensions."""
        c = self.config
        self.num_patches = (c.im_size // c.patch_size) ** 2
        self.projection_dim = c.num_channels * (c.patch_size**2)

        # Calculate dimensions per level
        self.encoder_dims = []
        self.encoder_patches = []
        self.encoder_hidden_dims = []

        for level in range(c.depth + 1):
            scale = 4**level
            dim = max(self.projection_dim // scale, 64)
            # dim = self.projection_dim // scale
            patches = self.num_patches * scale
            hidden = max(c.hidden_dim // (2**level), 256)

            self.encoder_dims.append(dim)
            self.encoder_patches.append(patches)
            self.encoder_hidden_dims.append(hidden)

    def _print_architecture_info(self):
        """Display architecture details."""
        print("ViT-UNet Architecture:")
        print(f"  Depth: {self.config.depth}")
        print(f"  Image size: {self.config.im_size}")
        print(f"  Patch size: {self.config.patch_size}")
        print(f"  Initial patches: {self.num_patches}")
        for i, (dim, patches) in enumerate(zip(self.encoder_dims, self.encoder_patches, strict=False)):
            print(f"  Level {i}: {patches} patches, dim={dim}")

    def _build_layers(self):
        """Construct all model layers."""
        self._build_patch_encoder()
        self._build_encoder()
        self._build_bottleneck()
        self._build_decoder()
        self._build_output()

    def _build_patch_encoder(self):
        """Create initial patch encoder."""
        c = self.config
        self.patch_encoder = PatchEncoder(
            c.im_size, c.patch_size, c.num_channels, self.encoder_dims[0], c.preprocessing
        )

    def _build_encoder(self):
        """Construct encoder transformer blocks."""
        c = self.config
        self.encoder_blocks = nn.ModuleList()

        for level in range(c.depth):
            level_blocks = nn.ModuleList(
                [
                    ReAttentionTransformerEncoder(
                        self.encoder_dims[level],
                        c.num_heads,
                        self.encoder_hidden_dims[level],
                        c.attn_drop,
                        c.proj_drop,
                        c.linear_drop,
                    )
                    for _ in range(c.depth_te)
                ]
            )
            self.encoder_blocks.append(level_blocks)

    def _build_bottleneck(self):
        """Create bottleneck transformer blocks."""
        c = self.config
        bottleneck_level = c.depth
        self.bottleneck = nn.ModuleList(
            [
                ReAttentionTransformerEncoder(
                    self.encoder_dims[bottleneck_level],
                    c.num_heads,
                    self.encoder_hidden_dims[bottleneck_level],
                    c.attn_drop,
                    c.proj_drop,
                    c.linear_drop,
                )
                for _ in range(c.size_bottleneck)
            ]
        )

    def _build_decoder(self):
        """Construct decoder transformer blocks and skip connections."""
        c = self.config
        self.decoder_blocks = nn.ModuleList()
        self.skip_connections = nn.ModuleList()

        for level in range(c.depth - 1, -1, -1):
            level_blocks = nn.ModuleList(
                [
                    ReAttentionTransformerEncoder(
                        self.encoder_dims[level],
                        c.num_heads,
                        self.encoder_hidden_dims[level],
                        c.attn_drop,
                        c.proj_drop,
                        c.linear_drop,
                    )
                    for _ in range(c.depth_te)
                ]
            )
            self.decoder_blocks.append(level_blocks)

            skip = SkipConnection(self.encoder_dims[level], c.num_heads, c.attn_drop)
            self.skip_connections.append(skip)

    def _build_output(self):
        """Create optional final convolution layer."""
        c = self.config
        if c.preprocessing == "conv":
            self.final_conv = nn.Conv2d(c.num_channels, c.num_channels, 3, padding=1)
        else:
            self.final_conv = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Main forward pass."""
        batch = x.shape[0]

        # Resize if needed
        if x.shape[-2:] != (self.config.im_size, self.config.im_size):
            x = F.interpolate(x, size=(self.config.im_size, self.config.im_size), mode="bilinear", align_corners=False)

        # Encode
        x_patch = self.patch_encoder(x)
        x_patch, encoder_skip = self._encode_patches(x_patch)

        # Bottleneck
        x_patch = self._process_bottleneck(x_patch)

        # Decode
        x_patch = self._decode_patches(x_patch, encoder_skip)

        # Reconstruct
        output = self._apply_final_processing(x_patch, batch, x)

        return output

    def _encode_patches(self, x_patch: torch.Tensor) -> list[torch.Tensor]:
        """Apply encoder blocks with downsampling."""
        encoder_skip = []

        for level, blocks in enumerate(self.encoder_blocks):
            # Apply transformer blocks
            for block in blocks:
                x_patch = block(x_patch)

            # Save for skip connection
            encoder_skip.append(x_patch)

            # Downsample
            # Use encoder_dims[level + 1] if needed
            x_patch = downsampling(x_patch, self.config.num_channels)

        return x_patch, encoder_skip

    def _process_bottleneck(self, x_patch: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck transformer blocks."""
        for block in self.bottleneck:
            x_patch = block(x_patch)
        return x_patch

    def _decode_patches(self, x_patch: torch.Tensor, encoder_skip: list[torch.Tensor]) -> torch.Tensor:
        """Apply decoder blocks with upsampling and skip connections."""
        for i, blocks in enumerate(self.decoder_blocks):
            # Apply skip connection
            x_patch = self._apply_skip_connection(x_patch, encoder_skip, i)

            # Apply transformer blocks
            for block in blocks:
                x_patch = block(x_patch)

        return x_patch

    def _apply_skip_connection(
        self, x_patch: torch.Tensor, encoder_skip: list[torch.Tensor], decoder_level: int
    ) -> torch.Tensor:
        """Upsample and apply skip connection."""
        # Upsample decoder features
        x_patch = upsampling(x_patch, self.config.num_channels)

        # Get corresponding encoder skip
        encoder_level = self.config.depth - 1 - decoder_level
        skip_feat = encoder_skip[encoder_level]

        # Validate shapes
        if x_patch.shape != skip_feat.shape:
            raise ValueError(f"Shape mismatch: {x_patch.shape} vs {skip_feat.shape}")

        # Apply skip connection
        x_patch = self.skip_connections[decoder_level](x_patch, skip_feat)

        return x_patch

    def _apply_final_processing(self, x_patch: torch.Tensor, batch: int, original_x: torch.Tensor) -> torch.Tensor:
        """Reconstruct image from patches."""
        # Unpatch
        output = unpatch(x_patch, self.config.num_channels)

        # Apply final convolution if enabled
        if self.final_conv is not None:
            output = self.final_conv(output)

        return output


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    # Create configuration
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
    output = model(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

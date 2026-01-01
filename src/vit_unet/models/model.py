from __future__ import annotations

from typing import Literal, TYPE_CHECKING

import numpy as np
import torch
import torchvision

if TYPE_CHECKING:
    from vit_unet.config.model_config import VitunetConfig

# 4: batch, channels, height, width
IMAGE_DIMS = 4
# 5: batch, n_patches, channels, height, width
PATCHED_IMAGE_DIMS = 5

MINIMUM_PATCH_SIZE = 4


# Auxiliary functions to create & undo patches
def patch(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    if len(x.size()) == PATCHED_IMAGE_DIMS:
        x = torch.squeeze(x, dim=1)
    height, width = x.shape[-2], x.shape[-1]
    if height % patch_size != 0 or width % patch_size != 0:
        msg = f"Patch size {patch_size} must divide image dimensions ({height}x{width})"
        raise ValueError(msg)
    patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patch_list = torch.flatten(patches, 2, 3).permute(0, 2, 1, 3, 4)
    return patch_list


def unflatten(flattened: torch.Tensor, num_channels: int) -> torch.Tensor:
    # Alberto: Added to reconstruct from batch, n_patches, projection_dim -> batch, n_patches, channels, height, width
    batch, n_patches, projection_dim = flattened.size()
    unflattened = torch.reshape(
        flattened,
        (
            batch,
            n_patches,
            num_channels,
            int(np.sqrt(projection_dim // num_channels)),
            int(np.sqrt(projection_dim // num_channels)),
        ),
    )
    return unflattened


def unpatch(x: torch.Tensor, num_channels: int) -> torch.Tensor:
    if len(x.size()) < PATCHED_IMAGE_DIMS:
        x = unflatten(x, num_channels)
    batch, n_patches, channels, height, width = x.size()
    if channels != num_channels:
        msg = "Number of channels does not match"
        raise ValueError(msg)
    elem_per_axis = int(np.sqrt(n_patches))

    # Reshape patches to grid layout and reconstruct image (vectorized, no loops)
    x_reshaped = x.reshape(batch, elem_per_axis, elem_per_axis, channels, height, width)
    # Transpose to get correct spatial arrangement
    x_transposed = x_reshaped.permute(0, 3, 1, 4, 2, 5)
    # Merge spatial dimensions
    restored_images = x_transposed.reshape(batch, 1, channels, height * elem_per_axis, width * elem_per_axis)
    return restored_images


# Auxiliary methods to downsampling & upsampling
def downsampling(encoded_patches: torch.Tensor, num_channels: int) -> torch.Tensor:
    """Downsample by reducing spatial resolution of patches."""
    _, _, projection_dim = encoded_patches.size()

    # Calculate current patch dimensions
    height = int(np.sqrt(projection_dim // num_channels))

    # Reconstruct image from patches
    x_unflat = unflatten(encoded_patches, num_channels)  # (batch, n_patches, channels, height, width)
    original_image = unpatch(x_unflat, num_channels)

    # Create new patches with smaller size (downsampling)
    new_patch_size = height // 2
    new_patches = patch(original_image, patch_size=new_patch_size)
    new_patches_flattened = torch.flatten(new_patches, start_dim=-3, end_dim=-1)
    return new_patches_flattened


def upsampling(encoded_patches: torch.Tensor, num_channels: int) -> torch.Tensor:
    """Upsample by increasing spatial resolution of patches."""
    _, _, projection_dim = encoded_patches.size()

    # Calculate current patch dimensions
    height = int(np.sqrt(projection_dim // num_channels))

    # Reconstruct image from patches
    x_unflat = unflatten(encoded_patches, num_channels)  # (batch, n_patches, channels, height, width)
    original_image = unpatch(x_unflat, num_channels)

    # Create new patches with larger size (upsampling)
    new_patch_size = height * 2
    new_patches = patch(original_image, patch_size=new_patch_size)
    new_patches_flattened = torch.flatten(new_patches, start_dim=-3, end_dim=-1)
    return new_patches_flattened


# Class PatchEncoder, to include initial and positional encoding
class PatchEncoder(torch.nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        num_channels: int,
        projection_dim: int | None = None,
        preprocessing: Literal["conv", "fourier", "none"] = "conv",
    ) -> None:
        super().__init__()
        # Validate preprocessing parameter
        if preprocessing not in {"conv", "fourier", "none"}:
            msg = f"preprocessing must be one of ['conv', 'fourier', 'none'], got '{preprocessing}'"
            raise ValueError(msg)
        # Parameters
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.num_channels = num_channels
        self.projection_dim = projection_dim or self.num_channels * self.patch_size**2
        self.preprocessing = preprocessing
        # Register positions as buffer so it moves with model to correct device
        self.register_buffer("positions", torch.arange(start=0, end=self.num_patches, step=1))

        # Layers
        if self.preprocessing == "conv":
            self.conv2d = torch.nn.Conv2d(self.num_channels, self.num_channels, 3, padding="same")
        self.position_embedding = torch.nn.Embedding(
            num_embeddings=self.num_patches,
            embedding_dim=self.projection_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.preprocessing == "conv":
            x = self.conv2d(x)
        elif self.preprocessing == "fourier":
            x = torch.fft.fft2(x).real  # pyright: ignore[reportUnknownVariableType]
        patches = patch(x, self.patch_size)  # pyright: ignore[reportUnknownArgumentType]
        flat_patches = torch.flatten(patches, -3, -1)
        # Add positional encoding directly (removed redundant unpatch-patch cycle)
        encoded = flat_patches + self.position_embedding(self.positions)
        return encoded


# AutoEncoder implementation
class FeedForward(torch.nn.Module):
    def __init__(self, projection_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(projection_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, projection_dim),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FformerEncoder(torch.nn.Module):
    def __init__(
        self,
        num_patches: int,
        projection_dim: int,
        hidden_dim: int,
        dropout: float,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.dtype = dtype
        # LayerNorm only on last dimension for flexibility
        self.LN = torch.nn.LayerNorm(self.projection_dim, dtype=self.dtype)
        self.FeedForward = FeedForward(
            projection_dim=self.projection_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            # dtype=self.dtype,
        )

    def forward(self, encoded_patches: torch.Tensor) -> torch.Tensor:
        # Removed FFT2 operation for stability
        encoded_patches = self.LN(encoded_patches)
        encoded_patches += self.FeedForward(encoded_patches)
        encoded_patches = self.LN(encoded_patches)
        return encoded_patches


class ReAttention(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        num_channels: int = 3,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        apply_transform: bool = True,
        transform_scale: bool = False,
    ) -> None:
        super().__init__()
        # Parameters
        self.num_heads = num_heads
        self.num_channels = num_channels
        head_dim = dim // num_heads
        self.apply_transform = apply_transform
        self.scale = qk_scale or head_dim**-0.5

        # Layers
        if apply_transform:
            self.reatten_matrix = torch.nn.Conv2d(self.num_heads, self.num_heads, 1, 1)
            self.var_norm = torch.nn.BatchNorm2d(self.num_heads)
            self.qconv2d = torch.nn.Conv2d(self.num_channels, self.num_channels, 3, padding="same", bias=qkv_bias)
            self.kconv2d = torch.nn.Conv2d(self.num_channels, self.num_channels, 3, padding="same", bias=qkv_bias)
            self.vconv2d = torch.nn.Conv2d(self.num_channels, self.num_channels, 3, padding="same", bias=qkv_bias)
            self.reatten_scale = self.scale if transform_scale else 1.0
        else:
            self.qconv2d = torch.nn.Conv2d(self.num_channels, self.num_channels, 3, padding="same", bias=qkv_bias)
            self.kconv2d = torch.nn.Conv2d(self.num_channels, self.num_channels, 3, padding="same", bias=qkv_bias)
            self.vconv2d = torch.nn.Conv2d(self.num_channels, self.num_channels, 3, padding="same", bias=qkv_bias)

        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)

    def _prepare_conv_input(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
        """Prepare input for convolution by unflattening and reshaping.

        Returns:
            tuple: (reshaped_input, batch, n_patches, projection_dim)
        """
        batch, n_patches, projection_dim = x.shape
        x_unflat = unflatten(x, self.num_channels)
        _, _, channels, height, width = x_unflat.shape
        x_conv_input = x_unflat.reshape(batch * n_patches, channels, height, width)
        return x_conv_input, batch, n_patches, projection_dim

    def _apply_qkv_convolutions(
        self, x_conv_input: torch.Tensor, batch: int, n_patches: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply Q, K, V convolutions and reshape back to patch format.

        Returns:
            tuple: (q_conv, k_conv, v_conv) each with shape (batch, n_patches, channels, height, width)
        """
        # Infer output shape from first convolution
        q_out = self.qconv2d(x_conv_input)
        _, channels, height, width = q_out.shape

        q_conv = q_out.reshape(batch, n_patches, channels, height, width)
        k_conv = self.kconv2d(x_conv_input).reshape(batch, n_patches, channels, height, width)
        v_conv = self.vconv2d(x_conv_input).reshape(batch, n_patches, channels, height, width)
        return q_conv, k_conv, v_conv

    def _prepare_attention_tensors(
        self,
        q_conv: torch.Tensor,
        k_conv: torch.Tensor,
        v_conv: torch.Tensor,
        batch: int,
        n_patches: int,
        projection_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Flatten and reshape convolution outputs for multi-head attention.

        Returns:
            tuple: (q, k, v) reshaped for attention computation
        """
        head_dim = projection_dim // self.num_heads

        q = torch.flatten(q_conv, -3, -1).reshape(batch, n_patches, self.num_heads, head_dim).transpose(1, 2)
        k = torch.flatten(k_conv, -3, -1).reshape(batch, n_patches, self.num_heads, head_dim).transpose(1, 2)
        v = torch.flatten(v_conv, -3, -1).reshape(batch, n_patches, self.num_heads, head_dim).transpose(1, 2)
        return q, k, v

    def _compute_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, batch: int, n_patches: int, projection_dim: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute multi-head attention and return output.

        Returns:
            tuple: (output, attention_weights)
        """
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.nn.functional.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        if self.apply_transform:
            attn = self.var_norm(self.reatten_matrix(attn)) * self.reatten_scale

        attn_weights = attn
        output = torch.matmul(attn, v).transpose(1, 2).reshape(batch, n_patches, projection_dim)
        output = self.proj(output)
        output = self.proj_drop(output)
        return output, attn_weights

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Prepare convolution input
        x_conv_input, batch, n_patches, projection_dim = self._prepare_conv_input(x)

        # Apply Q, K, V convolutions
        q_conv, k_conv, v_conv = self._apply_qkv_convolutions(x_conv_input, batch, n_patches)

        # Prepare attention tensors
        q, k, v = self._prepare_attention_tensors(q_conv, k_conv, v_conv, batch, n_patches, projection_dim)

        # Compute attention and get output
        output, attn_weights = self._compute_attention(q, k, v, batch, n_patches, projection_dim)

        return output, attn_weights


class ReAttentionTransformerEncoder(torch.nn.Module):
    def __init__(
        self,
        num_patches: int,
        num_channels: int,
        projection_dim: int,
        hidden_dim: int,
        num_heads: int,
        attn_drop: float,
        proj_drop: float,
        linear_drop: float,
    ) -> None:
        super().__init__()
        self.num_patches = num_patches
        self.num_channels = num_channels
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.linear_drop = linear_drop
        self.ReAttn = ReAttention(
            self.projection_dim,
            num_channels=self.num_channels,
            num_heads=self.num_heads,
            attn_drop=self.attn_drop,
            proj_drop=self.proj_drop,
        )
        # LayerNorm only on last dimension to support dynamic num_patches
        self.LN1 = torch.nn.LayerNorm(self.projection_dim)
        self.LN2 = torch.nn.LayerNorm(self.projection_dim)
        self.FeedForward = FeedForward(
            projection_dim=self.projection_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.linear_drop,
        )

    def forward(self, encoded_patches: torch.Tensor) -> torch.Tensor:
        encoded_patch_attn, _ = self.ReAttn(encoded_patches)
        encoded_patches = encoded_patch_attn + encoded_patches
        encoded_patches = self.LN1(encoded_patches)
        encoded_patches = self.FeedForward(encoded_patches) + encoded_patches
        encoded_patches = self.LN2(encoded_patches)
        return encoded_patches


# Skip connections
class SkipConnection(torch.nn.Module):
    """It is observed that similarity along same batch of data is extremely large.

    Thus can reduce the bs dimension when calculating the attention map.
    """

    def __init__(
        self,
        dim: int,
        num_channels: int = 3,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        transform_scale: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_channels = num_channels
        head_dim = dim // num_heads

        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim**-0.5
        self.reatten_matrix = torch.nn.Conv2d(self.num_heads, self.num_heads, 1, 1)
        self.var_norm = torch.nn.BatchNorm2d(self.num_heads)
        self.qconv2d = torch.nn.Conv2d(self.num_channels, self.num_channels, 3, padding="same", bias=qkv_bias)
        self.kconv2d = torch.nn.Conv2d(self.num_channels, self.num_channels, 3, padding="same", bias=qkv_bias)
        self.vconv2d = torch.nn.Conv2d(self.num_channels, self.num_channels, 3, padding="same", bias=qkv_bias)

        self.reatten_scale = self.scale if transform_scale else 1.0
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)

    def _prepare_qkv_inputs(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
        """Prepare Q, K, V inputs by unflattening and reshaping for convolution.

        Returns:
            tuple: (q_input, k_input, v_input, batch, n_patches, projection_dim)
        """
        batch, n_patches, projection_dim = q.shape

        q_unflat = unflatten(q, self.num_channels)
        k_unflat = unflatten(k, self.num_channels)
        v_unflat = unflatten(v, self.num_channels)

        _, _, channels, height, width = q_unflat.shape

        q_input = q_unflat.reshape(batch * n_patches, channels, height, width)
        k_input = k_unflat.reshape(batch * n_patches, channels, height, width)
        v_input = v_unflat.reshape(batch * n_patches, channels, height, width)

        return q_input, k_input, v_input, batch, n_patches, projection_dim

    def _apply_skip_convolutions(
        self, q_input: torch.Tensor, k_input: torch.Tensor, v_input: torch.Tensor, batch: int, n_patches: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply convolutions to Q, K, V for skip connection.

        Returns:
            tuple: (q_conv, k_conv, v_conv) reshaped to (batch, n_patches, channels, height, width)
        """
        q_out = self.qconv2d(q_input)
        _, channels, height, width = q_out.shape

        q_conv = q_out.reshape(batch, n_patches, channels, height, width)
        k_conv = self.kconv2d(k_input).reshape(batch, n_patches, channels, height, width)
        v_conv = self.vconv2d(v_input).reshape(batch, n_patches, channels, height, width)

        return q_conv, k_conv, v_conv

    def _compute_skip_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, batch: int, n_patches: int, projection_dim: int
    ) -> torch.Tensor:
        """Compute attention for skip connection and return output.

        Returns:
            torch.Tensor: Output after attention and projection
        """
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.nn.functional.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        attn = self.var_norm(self.reatten_matrix(attn)) * self.reatten_scale

        output = torch.matmul(attn, v).transpose(1, 2).reshape(batch, n_patches, projection_dim)
        output = self.proj(output)
        output = self.proj_drop(output)

        return output

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if q.shape != k.shape or k.shape != v.shape:
            msg = "Q, K, and V must have the same shape"
            raise ValueError(msg)

        # Prepare inputs
        q_input, k_input, v_input, batch, n_patches, projection_dim = self._prepare_qkv_inputs(q, k, v)

        # Apply convolutions
        q_conv, k_conv, v_conv = self._apply_skip_convolutions(q_input, k_input, v_input, batch, n_patches)

        # Prepare attention tensors (reuse method from ReAttention parent logic)
        head_dim = projection_dim // self.num_heads
        q_attn = torch.flatten(q_conv, -3, -1).reshape(batch, n_patches, self.num_heads, head_dim).transpose(1, 2)
        k_attn = torch.flatten(k_conv, -3, -1).reshape(batch, n_patches, self.num_heads, head_dim).transpose(1, 2)
        v_attn = torch.flatten(v_conv, -3, -1).reshape(batch, n_patches, self.num_heads, head_dim).transpose(1, 2)

        # Compute attention and output
        return self._compute_skip_attention(q_attn, k_attn, v_attn, batch, n_patches, projection_dim)


# Model architecture
class ViTUNet(torch.nn.Module):
    def __init__(self, config: VitunetConfig) -> None:
        super().__init__()
        self.config = config
        self._validate_config()
        self._setup_dimensions()
        self._print_architecture_info()
        self._build_layers()

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If depth is incompatible with patch size, final patch size is too small,
                       patch size doesn't divide image size, or preprocessing method is invalid.
        """
        final_patch_size = self.config.patch_size // (2**self.config.depth)
        if self.config.patch_size % (2**self.config.depth) != 0:
            msg = f"Depth {self.config.depth} incompatible: patch_size {self.config.patch_size} not divisible"
            raise ValueError(msg)
        if final_patch_size < MINIMUM_PATCH_SIZE:
            msg = f"Final patch size {final_patch_size} < minimum {MINIMUM_PATCH_SIZE}"
            raise ValueError(msg)
        if self.config.im_size % self.config.patch_size != 0:
            msg = f"Patch size {self.config.patch_size} must divide image size {self.config.im_size}"
            raise ValueError(msg)
        if self.config.preprocessing not in {"conv", "fourier", "none"}:
            msg = f"preprocessing must be 'conv', 'fourier', or 'none', got '{self.config.preprocessing}'"
            raise ValueError(msg)

    def _setup_dimensions(self) -> None:
        """Calculate derived dimensions from config."""
        self.depth = self.config.depth
        self.depth_te = self.config.depth_te
        self.size_bottleneck = self.config.size_bottleneck
        self.preprocessing = self.config.preprocessing
        self.im_size = self.config.im_size
        self.patch_size = self.config.patch_size
        self.num_patches = (self.im_size // self.patch_size) ** 2
        self.num_channels = self.config.num_channels
        self.projection_dim = self.num_channels * (self.patch_size) ** 2
        self.hidden_dim = self.config.hidden_dim
        self.num_heads = self.config.num_heads
        self.attn_drop = self.config.attn_drop
        self.proj_drop = self.config.proj_drop
        self.linear_drop = self.config.linear_drop
        self.verbose = self.config.verbose

    def _print_architecture_info(self) -> None:
        """Print architecture information."""
        print("Architecture information:")
        for i in range(self.config.depth + 1):
            print(f"Level {i}:")
            print("\tPatch size:", self.patch_size // (2**i))
            print("\tNum. patches:", self.num_patches * (4**i))
            print("\tProjection size:", (self.num_channels * self.patch_size**2) // (4**i))
            print("\tHidden dim. size:", self.hidden_dim // (2**i))

    def _build_layers(self) -> None:
        """Construct all model layers."""
        self._build_patch_encoder()
        self._build_encoder()
        self._build_bottleneck()
        self._build_decoder()
        self._build_output()

    def _build_patch_encoder(self) -> None:
        """Construct patch encoder."""
        self.PE = PatchEncoder(
            img_size=self.im_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            preprocessing=self.config.preprocessing,
        )

    def _build_encoder(self) -> None:
        """Construct encoder layers."""
        self.Encoders = torch.nn.ModuleList()
        for level in range(self.depth):
            exp_factor = 4 ** (level)
            exp_factor_hidden = 2 ** (level)
            for _ in range(self.depth_te):
                self.Encoders.append(
                    ReAttentionTransformerEncoder(
                        self.num_patches * exp_factor,
                        self.num_channels,
                        self.projection_dim // exp_factor,
                        self.hidden_dim // exp_factor_hidden,
                        self.num_heads,
                        self.attn_drop,
                        self.proj_drop,
                        self.linear_drop,
                    )
                )

    def _build_bottleneck(self) -> None:
        """Construct bottleneck layers."""
        self.BottleNeck = torch.nn.ModuleList()
        exp_factor = 4 ** (self.depth)
        exp_factor_hidden = 2 ** (self.depth)
        for _ in range(self.size_bottleneck):
            self.BottleNeck.append(
                ReAttentionTransformerEncoder(
                    self.num_patches * exp_factor,
                    self.num_channels,
                    self.projection_dim // exp_factor,
                    self.hidden_dim // exp_factor_hidden,
                    self.num_heads,
                    self.attn_drop,
                    self.proj_drop,
                    self.linear_drop,
                )
            )

    def _build_decoder(self) -> None:
        """Construct decoder layers and skip connections."""
        self.Decoders = torch.nn.ModuleList()
        self.SkipConnections = torch.nn.ModuleList()
        for level in range(self.depth):
            exp_factor = 4 ** (self.depth - level)
            exp_factor_skip = 4 ** (self.depth - level - 1)
            exp_factor_hidden = 2 ** (self.depth - level)
            for _ in range(self.depth_te):
                self.Decoders.append(
                    ReAttentionTransformerEncoder(
                        self.num_patches * exp_factor,
                        self.num_channels,
                        self.projection_dim // exp_factor,
                        self.hidden_dim // exp_factor_hidden,
                        self.num_heads,
                        self.attn_drop,
                        self.proj_drop,
                        self.linear_drop,
                    )
                )
            self.SkipConnections.append(
                SkipConnection(
                    dim=self.projection_dim // exp_factor_skip,
                    num_channels=self.num_channels,
                    num_heads=self.num_heads,
                    attn_drop=self.attn_drop,
                    proj_drop=self.proj_drop,
                )
            )

    def _build_output(self) -> None:
        """Construct output layer."""
        if self.preprocessing == "conv":
            self.conv2d = torch.nn.Conv2d(self.num_channels, self.num_channels, 3, padding="same")

    def _encode_patches(self, x_patch: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Process encoding layers and collect skip connections."""
        encoder_skip: list[torch.Tensor] = []
        for i, enc in enumerate(self.Encoders):
            x_patch = enc(x_patch)
            if (i + 1) % self.depth_te == 0:
                encoder_skip.append(x_patch)
                x_patch = downsampling(x_patch, self.num_channels)
                self._log_encoder_step(i, x_patch)
        return x_patch, encoder_skip

    def _process_bottleneck(self, x_patch: torch.Tensor) -> torch.Tensor:
        """Process bottleneck layers."""
        for i, bottle in enumerate(self.BottleNeck):
            x_patch = bottle(x_patch)
            self._log_bottleneck_step(i, x_patch)
        return x_patch

    def _decode_patches(self, x_patch: torch.Tensor, encoder_skip: list[torch.Tensor]) -> torch.Tensor:
        """Process decoding layers with skip connections."""
        for i, dec in enumerate(self.Decoders):
            x_patch = dec(x_patch)
            if (i + 1) % self.depth_te == 0:
                x_patch = self._apply_skip_connection(x_patch, encoder_skip, i)
                self._log_decoder_step(i, x_patch)
        return x_patch

    def _apply_skip_connection(self, x_patch: torch.Tensor, encoder_skip: list[torch.Tensor], i: int) -> torch.Tensor:
        """Apply upsampling and skip connection.

        Raises:
            ValueError: If encoder and decoder tensors have incompatible shapes.
        """
        x_patch = upsampling(x_patch, self.num_channels)
        skip_idx = self.depth - ((i + 1) // self.depth_te)
        if encoder_skip[skip_idx].shape != x_patch.shape:
            msg = "enc and dec not same shape"
            raise ValueError(msg)
        return self.SkipConnections[(i + 1) // self.depth_te - 1](encoder_skip[skip_idx], x_patch, x_patch)

    def _apply_final_processing(self, x_patch: torch.Tensor, batch: int, x: torch.Tensor) -> torch.Tensor:
        """Apply final preprocessing and return result."""
        x_restored = unpatch(unflatten(x_patch, self.num_channels), self.num_channels).reshape(
            batch, self.num_channels, self.im_size, self.im_size
        )

        if self.preprocessing == "conv":
            x_restored = self.conv2d(x_restored)
        elif self.preprocessing == "fourier":
            x_restored = torch.fft.ifft2(x, norm="ortho").real  # pyright: ignore[reportUnknownVariableType]

        self._log_final()
        return x_restored  # pyright: ignore[reportUnknownVariableType]

    def _log_encoder_step(self, i: int, x_patch: torch.Tensor) -> None:
        """Log encoder step if verbose."""
        if self.verbose:
            print(f"Encoder {i}")
            print("\t Shape after level " + str((i + 1) // self.depth_te) + " of encoding:", x_patch.size())
            print(torch.cuda.memory_summary("cuda"))

    def _log_bottleneck_step(self, i: int, x_patch: torch.Tensor) -> None:
        """Log bottleneck step if verbose."""
        if self.verbose:
            print(f"Bottleneck {i}")
            print("\tShape after step " + str(i + 1) + " of bottleneck:", x_patch.size())
            print(torch.cuda.memory_summary("cuda"))

    def _log_decoder_step(self, i: int, x_patch: torch.Tensor) -> None:
        """Log decoder step if verbose."""
        if self.verbose:
            print(f"Decoder {i}")
            print("\tShape after step " + str(i + 1) + " of decoder:", x_patch.size())
            print(torch.cuda.memory_summary("cuda"))

    def _log_final(self) -> None:
        """Log final step if verbose."""
        if self.verbose:
            print("Final")
            print(torch.cuda.memory_summary("cuda"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Previous validations
        x = torchvision.transforms.Resize(self.im_size)(x)
        batch, _, _, _ = x.size()

        # Preprocessing
        x_patch = self.PE(x)
        if self.verbose:
            print("Patch Encoder")
            print(torch.cuda.memory_summary("cuda"))

        # Encoding phase
        x_patch, encoder_skip = self._encode_patches(x_patch)

        # Bottleneck phase
        x_patch = self._process_bottleneck(x_patch)

        # Decoding phase
        x_patch = self._decode_patches(x_patch, encoder_skip)

        # Final processing
        return self._apply_final_processing(x_patch, batch, x)

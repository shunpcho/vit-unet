from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torchvision

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
    assert height % patch_size == 0, "Patch size must divide images height"
    assert width % patch_size == 0, "Patch size must divide images width"
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
    assert channels == num_channels, "Num. channels must agree"
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
    batch, n_patches, projection_dim = encoded_patches.size()

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
    batch, n_patches, projection_dim = encoded_patches.size()

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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, n_patches, projection_dim = x.shape

        # Unflatten once for efficiency
        x_unflat = unflatten(x, self.num_channels)  # (batch, n_patches, channels, height, width)
        batch, n_patches, channels, height, width = x_unflat.shape

        # Reshape to apply conv2d in batch: (batch*n_patches, channels, height, width)
        x_conv_input = x_unflat.reshape(batch * n_patches, channels, height, width)

        # Apply convolutions in batch (no loops)
        q_conv = self.qconv2d(x_conv_input).reshape(batch, n_patches, channels, height, width)
        k_conv = self.kconv2d(x_conv_input).reshape(batch, n_patches, channels, height, width)
        v_conv = self.vconv2d(x_conv_input).reshape(batch, n_patches, channels, height, width)

        # Flatten and reshape for attention
        q = (
            torch.flatten(q_conv, -3, -1)
            .reshape(batch, n_patches, self.num_heads, projection_dim // self.num_heads)
            .transpose(1, 2)
        )
        k = (
            torch.flatten(k_conv, -3, -1)
            .reshape(batch, n_patches, self.num_heads, projection_dim // self.num_heads)
            .transpose(1, 2)
        )
        v = (
            torch.flatten(v_conv, -3, -1)
            .reshape(batch, n_patches, self.num_heads, projection_dim // self.num_heads)
            .transpose(1, 2)
        )

        attn = (torch.matmul(q, k.transpose(-2, -1))) * self.scale
        attn = torch.nn.functional.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        if self.apply_transform:
            attn = self.var_norm(self.reatten_matrix(attn)) * self.reatten_scale
        attn_next = attn
        x = torch.matmul(attn, v).transpose(1, 2).reshape(batch, n_patches, projection_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_next


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

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        assert q.shape == k.shape
        assert k.shape == v.shape
        batch, n_patches, projection_dim = q.shape

        # Unflatten all inputs once
        q_unflat = unflatten(q, self.num_channels)  # (batch, n_patches, channels, height, width)
        k_unflat = unflatten(k, self.num_channels)
        v_unflat = unflatten(v, self.num_channels)
        batch, n_patches, channels, height, width = q_unflat.shape

        # Reshape to apply conv2d in batch: (batch*n_patches, channels, height, width)
        q_conv_input = q_unflat.reshape(batch * n_patches, channels, height, width)
        k_conv_input = k_unflat.reshape(batch * n_patches, channels, height, width)
        v_conv_input = v_unflat.reshape(batch * n_patches, channels, height, width)

        # Apply convolutions in batch (no loops)
        q_conv = self.qconv2d(q_conv_input).reshape(batch, n_patches, channels, height, width)
        k_conv = self.kconv2d(k_conv_input).reshape(batch, n_patches, channels, height, width)
        v_conv = self.vconv2d(v_conv_input).reshape(batch, n_patches, channels, height, width)

        # Flatten and reshape for attention
        q_attn = (
            torch.flatten(q_conv, -3, -1)
            .reshape(batch, n_patches, self.num_heads, projection_dim // self.num_heads)
            .transpose(1, 2)
        )
        k_attn = (
            torch.flatten(k_conv, -3, -1)
            .reshape(batch, n_patches, self.num_heads, projection_dim // self.num_heads)
            .transpose(1, 2)
        )
        v_attn = (
            torch.flatten(v_conv, -3, -1)
            .reshape(batch, n_patches, self.num_heads, projection_dim // self.num_heads)
            .transpose(1, 2)
        )

        attn = (torch.matmul(q_attn, k_attn.transpose(-2, -1))) * self.scale
        attn = torch.nn.functional.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        attn = self.var_norm(self.reatten_matrix(attn)) * self.reatten_scale

        x = torch.matmul(attn, v_attn).transpose(1, 2).reshape(batch, n_patches, projection_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# Model architecture
class ViTUNet(torch.nn.Module):
    def __init__(
        self,
        depth: int,
        depth_te: int,
        size_bottleneck: int,
        preprocessing: Literal["conv", "fourier", "none"],
        im_size: int,
        patch_size: int,
        num_channels: int,
        hidden_dim: int,
        num_heads: int,
        attn_drop: float,
        proj_drop: float,
        linear_drop: float,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        if patch_size % (2 ** (depth)) != 0:
            msg = "Depth must be adjusted, final patch size is incompatible."
            raise ValueError(msg)
        if patch_size // (2 ** (depth)) < MINIMUM_PATCH_SIZE:
            msg = f"Depth must be adjusted, final patch size is too small (lower than {MINIMUM_PATCH_SIZE})."
            raise ValueError(msg)
        if im_size % patch_size != 0:
            msg = "Patch size is not compatible with image size."
            raise ValueError(msg)
        if preprocessing not in {"conv", "fourier", "none"}:
            msg = f"preprocessing must be one of ['conv', 'fourier', 'none'], got '{preprocessing}'"
            raise ValueError(msg)
        # Parameters
        self.depth = depth
        self.depth_te = depth_te
        self.size_bottleneck = size_bottleneck
        self.preprocessing = preprocessing
        self.im_size = im_size
        self.patch_size = patch_size
        self.num_patches = (self.im_size // self.patch_size) ** 2
        self.num_channels = num_channels
        self.projection_dim = self.num_channels * (self.patch_size) ** 2
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.linear_drop = linear_drop
        self.verbose = verbose
        # Info
        print("Architecture information:")
        for i in range(depth + 1):
            print(f"Level {i}:")
            print("\tPatch size:", self.patch_size // (2**i))
            print("\tNum. patches:", self.num_patches * (4**i))
            print("\tProjection size:", (self.num_channels * self.patch_size**2) // (4**i))
            print("\tHidden dim. size:", self.hidden_dim // (2**i))
        # Layers
        self.PE = PatchEncoder(
            img_size=self.im_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            preprocessing=self.preprocessing,
        )

        self.Encoders = torch.nn.ModuleList()
        for level in range(self.depth):
            exp_factor = 4 ** (level)
            exp_factor_hidden = 2 ** (level)
            for _ in range(depth_te):
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
        self.BottleNeck = torch.nn.ModuleList()
        for _ in range(self.size_bottleneck):
            exp_factor = 4 ** (self.depth)
            exp_factor_hidden = 2 ** (self.depth)
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
        self.Decoders = torch.nn.ModuleList()
        self.SkipConnections = torch.nn.ModuleList()
        for level in range(self.depth):
            exp_factor = 4 ** (self.depth - level)
            exp_factor_skip = 4 ** (self.depth - level - 1)
            exp_factor_hidden = 2 ** (self.depth - level)
            for _ in range(depth_te):
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

        # Output
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

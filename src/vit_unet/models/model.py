from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torchvision

# 4: batch_size, c, h, w
IMAGE_DIMS = 4
# 5: batch_size, n_patches, c, h, w
PATCHED_IMAGE_DIMS = 5

MINIMUM_PATCH_SIZE = 4


# Auxiliary functions to create & undo patches
def patch(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    if len(x.size()) == PATCHED_IMAGE_DIMS:
        x = torch.squeeze(x, dim=1)
    h, w = x.shape[-2], x.shape[-1]
    assert h % patch_size == 0, "Patch size must divide images height"
    assert w % patch_size == 0, "Patch size must divide images width"
    patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patch_list = torch.flatten(patches, 2, 3).permute(0, 2, 1, 3, 4)
    return patch_list


def unflatten(flattened: torch.Tensor, num_channels: int) -> torch.Tensor:
    # Alberto: Added to reconstruct from bs, n, projection_dim -> bs, n, c, h, w
    bs, n, p = flattened.size()
    unflattened = torch.reshape(
        flattened, (bs, n, num_channels, int(np.sqrt(p // num_channels)), int(np.sqrt(p // num_channels)))
    )
    return unflattened


def unpatch(x: torch.Tensor, num_channels: int) -> torch.Tensor:
    if len(x.size()) < PATCHED_IMAGE_DIMS:
        batch_size, num_patches, ch, h, w = unflatten(x, num_channels).size()
    else:
        batch_size, num_patches, ch, h, w = x.size()
    assert ch == num_channels, "Num. channels must agree"
    elem_per_axis = int(np.sqrt(num_patches))
    patches_middle = torch.stack(
        [
            torch.cat(list(x.reshape(batch_size, elem_per_axis, elem_per_axis, ch, h, w)[i]), dim=-2)
            for i in range(batch_size)
        ],
        dim=0,
    )
    restored_images = torch.stack(
        [torch.cat(list(patches_middle[i]), dim=-1) for i in range(batch_size)], dim=0
    ).reshape(batch_size, 1, ch, h * elem_per_axis, w * elem_per_axis)
    return restored_images


# Auxiliary methods to downsampling & upsampling
def downsampling(encoded_patches: torch.Tensor, num_channels: int) -> torch.Tensor:
    _, _, embeddings = encoded_patches.size()
    # ch, h, w = num_channels, int(np.sqrt(embeddings / num_channels)), int(np.sqrt(embeddings / num_channels))
    h = int(np.sqrt(embeddings / num_channels))
    original_image = unpatch(unflatten(encoded_patches, num_channels), num_channels)
    new_patches = patch(original_image, patch_size=h // 2)
    new_patches_flattened = torch.flatten(new_patches, start_dim=-3, end_dim=-1)
    return new_patches_flattened


def upsampling(encoded_patches: torch.Tensor, num_channels: int) -> torch.Tensor:
    _, _, embeddings = encoded_patches.size()
    h = int(np.sqrt(embeddings / num_channels))
    original_image = unpatch(unflatten(encoded_patches, num_channels), num_channels)
    new_patches = patch(original_image, patch_size=h * 2)
    new_patches_flattened = torch.flatten(new_patches, start_dim=-3, end_dim=-1)
    return new_patches_flattened


class PatchEncoder(torch.nn.Module):
    """Patch encoder with optional preprocessing and positional encoding.

    Converts images into patches and applies positional embeddings.
    Supports convolutional, Fourier, or no preprocessing.
    """
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
        self.positions = torch.arange(
            start=0, end=self.num_patches, step=1, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )

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
            x = torch.fft.fft2(x).real
        patches = patch(x, self.patch_size)
        flat_patches = torch.flatten(patches, -3, -1)
        encoded = flat_patches + self.position_embedding(self.positions)
        encoded = torch.flatten(
            patch(unpatch(unflatten(encoded, self.num_channels), self.num_channels), patch_size=self.patch_size), -3, -1
        )
        return encoded


class FeedForward(torch.nn.Module):
    """Feed-forward network with GELU activation and dropout."""

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
        self.LN = torch.nn.LayerNorm(
            normalized_shape=[self.num_patches, self.projection_dim],
            dtype=self.dtype,
        )
        self.FeedForward = FeedForward(
            projection_dim=self.projection_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            # dtype=self.dtype,
        )

    def forward(self, encoded_patches: torch.Tensor) -> torch.Tensor:
        encoded_patches += torch.fft.fft2(encoded_patches).real
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
        batch_num, n_patches, channels = x.shape
        q = (
            torch.flatten(torch.stack([self.qconv2d(y) for y in unflatten(x, self.num_channels)], dim=0), -3, -1)
            .reshape(batch_num, n_patches, 1, self.num_heads, channels // self.num_heads)
            .permute(2, 0, 3, 1, 4)[0]
        )
        k = (
            torch.flatten(torch.stack([self.kconv2d(y) for y in unflatten(x, self.num_channels)], dim=0), -3, -1)
            .reshape(batch_num, n_patches, 1, self.num_heads, channels // self.num_heads)
            .permute(2, 0, 3, 1, 4)[0]
        )
        v = (
            torch.flatten(torch.stack([self.vconv2d(y) for y in unflatten(x, self.num_channels)], dim=0), -3, -1)
            .reshape(batch_num, n_patches, 1, self.num_heads, channels // self.num_heads)
            .permute(2, 0, 3, 1, 4)[0]
        )
        attn = (torch.matmul(q, k.transpose(-2, -1))) * self.scale
        attn = torch.nn.functional.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        if self.apply_transform:
            attn = self.var_norm(self.reatten_matrix(attn)) * self.reatten_scale
        attn_next = attn
        x = (torch.matmul(attn, v)).transpose(1, 2).reshape(batch_num, n_patches, channels)
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
        self.LN1 = torch.nn.LayerNorm(
            normalized_shape=[self.num_patches, self.projection_dim],
        )
        self.LN2 = torch.nn.LayerNorm(
            normalized_shape=[self.num_patches, self.projection_dim],
        )
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
        batch_num, n_patches, channels = q.shape
        q = (
            torch.flatten(torch.stack([self.qconv2d(y) for y in unflatten(q, self.num_channels)], dim=0), -3, -1)
            .reshape(batch_num, n_patches, 1, self.num_heads, channels // self.num_heads)
            .permute(2, 0, 3, 1, 4)[0]
        )
        k = (
            torch.flatten(torch.stack([self.kconv2d(y) for y in unflatten(k, self.num_channels)], dim=0), -3, -1)
            .reshape(batch_num, n_patches, 1, self.num_heads, channels // self.num_heads)
            .permute(2, 0, 3, 1, 4)[0]
        )
        v = (
            torch.flatten(torch.stack([self.vconv2d(y) for y in unflatten(v, self.num_channels)], dim=0), -3, -1)
            .reshape(batch_num, n_patches, 1, self.num_heads, channels // self.num_heads)
            .permute(2, 0, 3, 1, 4)[0]
        )
        attn = (torch.matmul(q, k.transpose(-2, -1))) * self.scale
        attn = torch.nn.functional.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        attn = self.var_norm(self.reatten_matrix(attn)) * self.reatten_scale

        x = torch.matmul(attn, v).transpose(1, 2).reshape(batch_num, n_patches, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class HViT_UNet(torch.nn.Module):
    """Hierarchical Vision Transformer U-Net for image denoising and restoration.

    A U-Net architecture using vision transformers with re-attention mechanisms
    and skip connections for efficient image processing.
    """

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Previous validations
        x = torchvision.transforms.Resize(self.im_size)(x)
        batch_size, _, _, _ = x.size()

        # "Preprocessing"
        x_patch = self.PE(x)
        if self.verbose:
            print("Patch Encoder")
            print(torch.cuda.memory_summary("cuda"))

        # Encoders
        encoder_skip = []
        for i, enc in enumerate(self.Encoders):
            x_patch = enc(x_patch)
            if (i + 1) % self.depth_te == 0:
                encoder_skip.append(x_patch)
                x_patch = downsampling(x_patch, self.num_channels)
                if self.verbose:
                    print(f"Encoder {i}")
                    print("\t Shape after level " + str((i + 1) // self.depth_te) + " of encoding:", x_patch.size())
                    print(torch.cuda.memory_summary("cuda"))

        # Bottleneck
        for i, bottle in enumerate(self.BottleNeck):
            x_patch = bottle(x_patch)
            if self.verbose:
                print(f"Bottleneck {i}")
                print("\tShape after step " + str(i + 1) + " of bottleneck:", x_patch.size())
                print(torch.cuda.memory_summary("cuda"))

        # Decoders
        for i, dec in enumerate(self.Decoders):
            x_patch = dec(x_patch)
            if (i + 1) % self.depth_te == 0:
                x_patch = upsampling(x_patch, self.num_channels)
                assert encoder_skip[self.depth - ((i + 1) // self.depth_te)].shape == x_patch.shape, (
                    "enc and dec not same shape"
                )
                x_patch = self.SkipConnections[(i + 1) // self.depth_te - 1](
                    encoder_skip[self.depth - ((i + 1) // self.depth_te)], x_patch, x_patch
                )
                if self.verbose:
                    print(f"Decoder {i}")
                    print("\tShape after step " + str(i + 1) + " of decoder:", x_patch.size())
                    print(torch.cuda.memory_summary("cuda"))

        # Output
        x_restored = unpatch(unflatten(x_patch, self.num_channels), self.num_channels).reshape(
            batch_size, self.num_channels, self.im_size, self.im_size
        )
        if self.preprocessing == "conv":
            x_restored = self.conv2d(x_restored)
        elif self.preprocessing == "fourier":
            x_restored = torch.fft.ifft2(x, norm="ortho").real
        if self.verbose:
            print("Final")
            print(torch.cuda.memory_summary("cuda"))

        return x_restored


def get_vit_unet(model_string: str, verbose: bool = False) -> HViT_UNet:
    """Factory function to create a pre-configured HViT_UNet model.

    Args:
        model_string: Model size variant ('lite', 'base', or 'large').
        verbose: Whether to print verbose debug information during forward pass.

    Returns:
        Configured HViT_UNet model instance.

    Raises:
        ValueError: If model_string is not a valid variant.
    """
    if model_string.lower() == "lite":
        return HViT_UNet(
            depth=2,
            depth_te=1,
            size_bottleneck=2,
            preprocessing="conv",
            im_size=224,
            patch_size=16,
            num_channels=3,
            hidden_dim=64,
            num_heads=4,
            attn_drop=0.2,
            proj_drop=0.2,
            linear_drop=0,
            verbose=verbose,
        )

    if model_string.lower() == "base":
        return HViT_UNet(
            depth=2,
            depth_te=2,
            size_bottleneck=2,
            preprocessing="conv",
            im_size=224,
            patch_size=32,
            num_channels=3,
            hidden_dim=128,
            num_heads=8,
            attn_drop=0.2,
            proj_drop=0.2,
            linear_drop=0,
            verbose=verbose,
        )

    if model_string.lower() == "large":
        return HViT_UNet(
            depth=2,
            depth_te=4,
            size_bottleneck=4,
            preprocessing="conv",
            im_size=224,
            patch_size=32,
            num_channels=3,
            hidden_dim=128,
            num_heads=8,
            attn_drop=0.2,
            proj_drop=0.2,
            linear_drop=0,
            verbose=verbose,
        )
    msg = f"Model string {model_string} not valid"
    raise ValueError(msg)

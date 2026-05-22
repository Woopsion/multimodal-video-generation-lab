"""Day 8: minimal MiniViT forward shape demo.

This script turns the Day 8 patch-embedding steps into an nn.Module:

image
-> patchify
-> patch projection
-> CLS token
-> position embedding
-> one Transformer encoder block
-> CLS image representation

It does not train a model and does not include a classifier head.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class MiniViTConfig:
    image_height: int = 4
    image_width: int = 4
    patch_height: int = 2
    patch_width: int = 2
    in_channels: int = 1
    embed_dim: int = 6
    num_heads: int = 2
    mlp_dim: int = 12


class MiniViT(nn.Module):
    def __init__(self, config: MiniViTConfig) -> None:
        super().__init__()
        if config.image_height % config.patch_height != 0:
            raise ValueError("image_height must be divisible by patch_height")
        if config.image_width % config.patch_width != 0:
            raise ValueError("image_width must be divisible by patch_width")
        if config.embed_dim % config.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.config = config
        self.patch_rows = config.image_height // config.patch_height
        self.patch_cols = config.image_width // config.patch_width
        self.num_patches = self.patch_rows * self.patch_cols
        self.patch_dim = config.in_channels * config.patch_height * config.patch_width

        self.patch_projection = nn.Linear(self.patch_dim, config.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, 1 + self.num_patches, config.embed_dim)
        )
        self.encoder_block = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.mlp_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        batch_size = images.shape[0]

        patches_6d = images.unfold(2, cfg.patch_height, cfg.patch_height).unfold(
            3, cfg.patch_width, cfg.patch_width
        )
        patches_ordered = patches_6d.permute(0, 2, 3, 1, 4, 5).contiguous()
        return patches_ordered.reshape(batch_size, self.num_patches, self.patch_dim)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = images.shape[0]

        patches = self.patchify(images)
        patch_embeddings = self.patch_projection(patches)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, patch_embeddings], dim=1)
        vit_input = tokens + self.position_embeddings
        encoder_output = self.encoder_block(vit_input)
        cls_output = encoder_output[:, 0, :]

        return {
            "patches": patches,
            "patch_embeddings": patch_embeddings,
            "cls_tokens": cls_tokens,
            "tokens": tokens,
            "vit_input": vit_input,
            "encoder_output": encoder_output,
            "cls_output": cls_output,
        }


def section(title: str) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


def show_shape(name: str, tensor: torch.Tensor, note: str) -> None:
    print(f"{name}.shape = {tuple(tensor.shape)}")
    print(f"{name}.dtype = {tensor.dtype}")
    print(note)


def main() -> None:
    torch.manual_seed(42)
    torch.set_printoptions(precision=4, sci_mode=False)

    config = MiniViTConfig()
    model = MiniViT(config)
    model.eval()

    section("Step 01: inspect MiniViT config")
    print(config)
    print(f"patch_rows = {model.patch_rows}")
    print(f"patch_cols = {model.patch_cols}")
    print(f"num_patches = {model.num_patches}")
    print(f"patch_dim = {model.patch_dim}")
    print(f"sequence_length_with_cls = {1 + model.num_patches}")

    section("Step 02: inspect learnable parameters")
    for name, parameter in model.named_parameters():
        print(f"{name:42s} shape = {tuple(parameter.shape)}")

    section("Step 03: run one forward pass")
    image = torch.arange(1, 17, dtype=torch.float32).reshape(1, 1, 4, 4)
    show_shape("image", image, "[B, C, H, W]: one tiny grayscale image.")

    with torch.no_grad():
        outputs = model(image)

    for name, tensor in outputs.items():
        notes = {
            "patches": "[B, num_patches, patch_dim]: flattened patch rows.",
            "patch_embeddings": "[B, num_patches, embed_dim]: patch vectors after Linear.",
            "cls_tokens": "[B, 1, embed_dim]: learnable summary token copied for batch.",
            "tokens": "[B, 1 + num_patches, embed_dim]: CLS plus patch tokens.",
            "vit_input": "[B, 1 + num_patches, embed_dim]: tokens plus position embeddings.",
            "encoder_output": "[B, 1 + num_patches, embed_dim]: output from one encoder block.",
            "cls_output": "[B, embed_dim]: first token used as image representation.",
        }
        show_shape(name, tensor, notes[name])

    section("Summary")
    print("MiniViT.forward(image) returns the same Day 8 chain packaged as a model:")
    print("image -> patches -> patch_embeddings -> tokens -> vit_input -> encoder_output -> cls_output")


if __name__ == "__main__":
    main()

"""Day 8: ViT patch embedding shape demo.

This script does not train a model. It only makes the shape flow visible:

image [B, C, H, W]
-> unfold patches
-> flatten patches
-> Linear patch projection
-> prepend CLS token
-> add position embedding
-> ViT input [B, 1 + num_patches, embed_dim]
-> one Transformer encoder block
-> encoder output [B, 1 + num_patches, embed_dim]
"""

from __future__ import annotations

import torch
from torch import nn


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

    section("Step 01: create a tiny image tensor")
    image = torch.arange(1, 17, dtype=torch.float32).reshape(1, 1, 4, 4)
    show_shape("image", image, "[B, C, H, W] = [1 image, 1 channel, 4 height, 4 width]")
    print("image[0, 0] =")
    print(image[0, 0])

    batch_size, channels, height, width = image.shape
    patch_h = 2
    patch_w = 2
    stride_h = 2
    stride_w = 2

    section("Step 02: unfold height first, then width")
    after_height = image.unfold(2, patch_h, stride_h)
    show_shape(
        "after_height",
        after_height,
        "[B, C, patch_rows, W, patch_h]: height has been windowed, width is not cut yet.",
    )

    patches_6d = after_height.unfold(3, patch_w, stride_w)
    show_shape(
        "patches_6d",
        patches_6d,
        "[B, C, patch_rows, patch_cols, patch_h, patch_w]: height and width are both cut.",
    )
    print("patches_6d[0, 0, 0, 0]  # top-left patch")
    print(patches_6d[0, 0, 0, 0])
    print("patches_6d[0, 0, 0, 1]  # top-right patch")
    print(patches_6d[0, 0, 0, 1])
    print("patches_6d[0, 0, 1, 0]  # bottom-left patch")
    print(patches_6d[0, 0, 1, 0])
    print("patches_6d[0, 0, 1, 1]  # bottom-right patch")
    print(patches_6d[0, 0, 1, 1])

    section("Step 03: reorder dimensions and flatten each patch")
    patches_ordered = patches_6d.permute(0, 2, 3, 1, 4, 5)
    show_shape(
        "patches_ordered",
        patches_ordered,
        "[B, patch_rows, patch_cols, C, patch_h, patch_w]: patch grid comes before patch content.",
    )

    patch_rows = height // patch_h
    patch_cols = width // patch_w
    num_patches = patch_rows * patch_cols
    patch_dim = channels * patch_h * patch_w
    patches = patches_ordered.reshape(batch_size, num_patches, patch_dim)
    show_shape(
        "patches",
        patches,
        "[B, num_patches, patch_dim]: every patch is now one flat row.",
    )
    print("patches[0] =")
    print(patches[0])

    section("Step 04: project flattened patches into embedding vectors")
    embed_dim = 6
    patch_projection = nn.Linear(in_features=patch_dim, out_features=embed_dim)

    with torch.no_grad():
        patch_projection.weight.copy_(
            torch.tensor(
                [
                    [0.10, 0.00, 0.00, 0.00],
                    [0.00, 0.10, 0.00, 0.00],
                    [0.00, 0.00, 0.10, 0.00],
                    [0.00, 0.00, 0.00, 0.10],
                    [0.05, 0.05, 0.05, 0.05],
                    [-0.05, 0.05, -0.05, 0.05],
                ]
            )
        )
        patch_projection.bias.zero_()

    print(f"patch_projection.weight.shape = {tuple(patch_projection.weight.shape)}")
    print("[embed_dim, patch_dim]: every output dimension has one row of weights.")
    print(f"patch_projection.bias.shape = {tuple(patch_projection.bias.shape)}")

    patch_embeddings = patch_projection(patches)
    show_shape(
        "patch_embeddings",
        patch_embeddings,
        "[B, num_patches, embed_dim]: Linear changes only the last dimension.",
    )
    print("patch_embeddings[0, 0]  # embedding vector for patch 0")
    print(patch_embeddings[0, 0])

    section("Step 05: prepend a learnable CLS token")
    cls_token = nn.Parameter(torch.full((1, 1, embed_dim), 0.5))
    cls_tokens = cls_token.expand(batch_size, -1, -1)
    show_shape("cls_token", cls_token, "[1, 1, embed_dim]: one learnable summary token template.")
    show_shape("cls_tokens", cls_tokens, "[B, 1, embed_dim]: expanded for the current batch.")

    tokens = torch.cat([cls_tokens, patch_embeddings], dim=1)
    show_shape(
        "tokens",
        tokens,
        "[B, 1 + num_patches, embed_dim]: CLS is placed before patch tokens.",
    )

    section("Step 06: add learnable position embeddings")
    position_embeddings = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
    with torch.no_grad():
        for position_index in range(1 + num_patches):
            position_embeddings[0, position_index] = position_index / 10.0

    show_shape(
        "position_embeddings",
        position_embeddings,
        "[1, 1 + num_patches, embed_dim]: one position vector for CLS plus each patch.",
    )
    print("position_embeddings[0, :, 0]  # first number of each position vector")
    print(position_embeddings[0, :, 0])

    vit_input = tokens + position_embeddings
    show_shape(
        "vit_input",
        vit_input,
        "[B, 1 + num_patches, embed_dim]: ready to enter a ViT encoder block.",
    )
    print("vit_input[0, 0]  # CLS vector after adding position 0")
    print(vit_input[0, 0])
    print("vit_input[0, 1]  # patch 0 vector after adding position 1")
    print(vit_input[0, 1])

    section("Step 07: pass ViT input through one Transformer encoder block")
    num_heads = 2
    dim_feedforward = 12
    encoder_block = nn.TransformerEncoderLayer(
        d_model=embed_dim,
        nhead=num_heads,
        dim_feedforward=dim_feedforward,
        dropout=0.0,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    encoder_block.eval()

    print(f"encoder_block.self_attn.num_heads = {encoder_block.self_attn.num_heads}")
    print(f"head_dim = embed_dim // num_heads = {embed_dim} // {num_heads} = {embed_dim // num_heads}")
    print(f"self_attn.in_proj_weight.shape = {tuple(encoder_block.self_attn.in_proj_weight.shape)}")
    print(f"self_attn.out_proj.weight.shape = {tuple(encoder_block.self_attn.out_proj.weight.shape)}")
    print(f"linear1.weight.shape = {tuple(encoder_block.linear1.weight.shape)}")
    print(f"linear2.weight.shape = {tuple(encoder_block.linear2.weight.shape)}")

    with torch.no_grad():
        encoder_output = encoder_block(vit_input)

    show_shape(
        "encoder_output",
        encoder_output,
        "[B, 1 + num_patches, embed_dim]: encoder mixes tokens but keeps the same shape.",
    )
    print("encoder_output[0, 0]  # CLS output after seeing all patch tokens")
    print(encoder_output[0, 0])

    cls_output = encoder_output[:, 0, :]
    show_shape(
        "cls_output",
        cls_output,
        "[B, embed_dim]: the first token can be used as the image-level representation.",
    )

    section("Summary")
    print("image -> patches -> patch embeddings -> CLS + positions -> ViT input")
    print(
        f"[1, 1, 4, 4] -> [1, {num_patches}, {patch_dim}] -> "
        f"[1, {num_patches}, {embed_dim}] -> [1, {1 + num_patches}, {embed_dim}] "
        f"-> [1, {1 + num_patches}, {embed_dim}]"
    )


if __name__ == "__main__":
    main()

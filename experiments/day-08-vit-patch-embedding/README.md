# Day 8 - ViT Patch Embedding

This folder contains the Day 8 minimal shape demo for Vision Transformer input construction.

Run:

```bash
.venv/bin/python experiments/day-08-vit-patch-embedding/vit_patch_shape_demo.py
```

The script does not train a model. It only prints the shape flow:

```text
image [B, C, H, W]
-> unfold patches
-> flatten patches
-> Linear patch projection
-> prepend CLS token
-> add position embedding
-> ViT input [B, 1 + num_patches, embed_dim]
```

Run the same idea packaged as an `nn.Module`:

```bash
.venv/bin/python experiments/day-08-vit-patch-embedding/mini_vit_shape_demo.py
```

The MiniViT demo still does not train a model. It only shows how a real model
class can organize the same forward path.

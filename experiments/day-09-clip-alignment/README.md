# Day 9 - CLIP Alignment

This folder contains the Day 9 minimal CLIP similarity and contrastive loss demo.

Run:

```bash
.venv/bin/python experiments/day-09-clip-alignment/clip_similarity_demo.py
```

The script does not load a real CLIP model and does not train anything. It only
prints the core shape flow:

```text
image_embeddings [B, D]
text_embeddings  [B, D]
-> logits = image_embeddings @ text_embeddings.T  [B, B]
-> labels = [0, 1, ..., B - 1]
-> image-to-text CrossEntropyLoss
-> text-to-image CrossEntropyLoss
-> final CLIP loss
```

It also runs one `backward()` call to show that gradients flow back through both
the image side and text side.

Run the real Transformers CLIP inference demo:

```bash
.venv/bin/python experiments/day-09-clip-alignment/clip_real_inference_demo.py
```

This second script is offline/local-only by default. If the CLIP model is not
already cached locally, run with explicit download permission:

```bash
.venv/bin/python experiments/day-09-clip-alignment/clip_real_inference_demo.py --allow-download
```

It creates a simple red-square image in memory, compares it with candidate text
prompts, and prints `pixel_values`, `input_ids`, `image_embeds`, `text_embeds`,
`logits_per_image`, and the final text ranking.

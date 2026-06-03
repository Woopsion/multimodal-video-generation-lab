# Day 10 - CLIP Retrieval

This folder contains the Day 10 CLIP-style retrieval demos.

Run the toy retrieval demo:

```bash
.venv/bin/python experiments/day-10-clip-retrieval/clip_retrieval_toy_demo.py
```

The toy script does not load a real CLIP model and does not train anything. It
uses hand-written image/text vectors to show the retrieval mechanics:

```text
image_embeddings [num_images, D]
text_embeddings  [num_texts, D]
-> scores = image_embeddings @ text_embeddings.T
-> image-to-text top-k retrieval
-> text-to-image top-k retrieval
-> Recall@K
```

Run the real Transformers CLIP retrieval demo:

```bash
.venv/bin/python experiments/day-10-clip-retrieval/clip_real_retrieval_demo.py
```

The real script is offline/local-only by default. If the CLIP model is not
already cached locally, run with explicit download permission:

```bash
.venv/bin/python experiments/day-10-clip-retrieval/clip_real_retrieval_demo.py --allow-download
```

The default real demo creates 4 generated images and 7 candidate texts. The
first 4 texts have matching images; the last 3 are no-answer queries used to
study threshold-based rejection.

Final Day 10 web demo:

```bash
.venv/bin/python demos/clip-image-search/app.py
```

Then open:

```text
http://127.0.0.1:8010
```

The web demo wraps the same retrieval workflow into a browser UI:
text query -> CLIP logits -> top-k ranking -> threshold accept/reject.

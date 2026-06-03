# Day 10 - CLIP Image Search Demo

This is the final Day 10 mini demo: a local web app that turns the CLIP
retrieval script into a small image-search tool.

Run:

```bash
.venv/bin/python demos/clip-image-search/app.py
```

Then open:

```text
http://127.0.0.1:8010
```

If port `8010` is already in use, either open the existing app at the URL
above, or run another instance on a different port:

```bash
.venv/bin/python demos/clip-image-search/app.py --port 8011
```

Stop the server with `Ctrl+C` in the terminal that started it.

The app is offline/local-only by default. If the CLIP model is not cached
locally, run with explicit download permission:

```bash
.venv/bin/python demos/clip-image-search/app.py --allow-download
```

Flow:

```text
browser query
-> /api/search
-> CLIPModel forward
-> logits_per_text [1, num_images]
-> top-k ranking
-> threshold accept/reject
```

The default gallery contains four generated images. Preset queries include
both matching prompts and no-answer prompts so threshold rejection is visible.

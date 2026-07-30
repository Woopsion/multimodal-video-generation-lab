# multimodal-video-generation-lab

A repository for learning, paper reading, engineering practice, and experimental records related to multimodal large models and video generation models.

## Current Learning Entry Points

- Plan: `multimodal-video-generation-learning-plan.md`
- Executable checklist: `docs/30-day-executable-checklist.md`
- Daily check-in log: `logs/daily-checkin.md`
- Week 1 review: `docs/week-01-foundation-review/index.html`
- Active Week 2 note: `docs/week-02-multimodal-understanding/index.html`
- Notes: `notes/`
- Experiments: `experiments/`

## Current Completed Milestone

Week 1 foundation study has been completed. The repo now contains runnable PyTorch, CNN, Transformer Encoder, and LLM inference learning demos with HTML notes.

- Day 5 final note: `docs/day-05-transformer-encoder/index.html`
- Day 6 final note: `docs/day-06-llm-inference/index.html`
- Week 1 review: `docs/week-01-foundation-review/index.html`
- Week 1 status note: `notes/week-01-foundation.md`

## Current Learning Focus

Week 2 has started. The learning format continues the HTML-first pattern: concepts are explained visually first, and code demos are added only after the shape and data flow are clear.

- Active note: `docs/week-02-multimodal-understanding/index.html`
- Day 8 final note: `docs/day-08-vit-patch-embedding/index.html`
- Day 9 final note: `docs/day-09-clip-alignment/index.html`
- Day 10 final note: `docs/day-10-clip-retrieval/index.html`
- Day 10 demos: `experiments/day-10-clip-retrieval/`, `demos/clip-image-search/`
- Day 11 final note: `docs/day-11-clip-paper-reading/index.html`
- Day 11 paper note: `papers/clip.md`
- Day 12 final note: `docs/day-12-vlm-image-qa/index.html`
- Day 12 status note: `notes/day-12-vlm-image-qa.md`
- Day 12 VLM QA demo: `experiments/vlm-qa/image_qa.py`
- Day 13 final note: `docs/day-13-llava-paper-reading/index.html`
- Day 13 status note: `notes/day-13-llava-paper-reading.md`
- Day 13 paper note: `papers/llava.md`
- Day 14 final note: `docs/day-14-week2-review-vlm-assistant/index.html`
- Day 14 status note: `notes/day-14-week2-review-vlm-assistant.md`
- Day 14 MVP design: `demos/vlm-visual-assistant/design.md`
- Day 15 active note: `docs/day-15-diffusion-basics/index.html`
- Week 3 status note: `notes/week-03-diffusion.md`
- Status note: `notes/week-02-vlm.md`
- Current task: Day 15 Step 03, understanding DDPM reverse denoising and the division of work between the noise predictor and scheduler.

## Repository Hygiene

- Local datasets under `experiments/*/data/` are ignored.
- Local pretrained model folders under `experiments/*/pretrained/` are ignored.
- Generated `.pt` checkpoints under `experiments/*/outputs/` are ignored for future runs.
- Day 6 real Transformers demos are offline/local-only by default and require `--allow-download` before downloading Hugging Face model files.

# P3 Temporal Embedding Diagnostics

CPU-side preparation for a kill-test on temporal padding with frozen InternVideo2.

## Goal
Smoke-test whether exact text-to-video Recall@1 degrades under three frame conditions (budget=8, k=4):

- **clean**: 8 evenly spaced frames from the clip
- **pad_dup**: 4 kept frames; first 4 slots are duplicates of the kept frames
- **pad_junk**: 4 kept frames; first 4 slots are frames from a different clip

Same kept indices for every condition of a given clip.

## Status
- Frame sampler written (`scripts/frame_sampler.py`)
- Clip source search pending / blocked (no local clips, free short clips with one-line captions not yet materialized into `data/`)
- Manifest not yet written
- Do not train. Frozen InternVideo2 only.

## Usage (GPU machine)
```
cd p3-temporal
git pull
export HF_HOME="$PWD/hf-cache"
export HUGGINGFACE_HUB_CACHE="$PWD/hf-cache/hub"
python scripts/smoke_recall.py   # once manifest + clips exist
```

# P3 — early-frame bias in video-text dual encoders (Block 1 kill test)

Does text-to-video retrieval depend on *where* junk frames sit (start/middle/end) at a fixed 8-frame budget,
beyond plain dilution and beyond a position-free mean-pooled image-CLIP null?

## Data (not in repo)
HF dataset `friedrichor/MSR-VTT`: `MSRVTT_Videos.zip`, `msrvtt_test_1k.json` (1k-A test, 1 caption/clip).
```
hf download friedrichor/MSR-VTT MSRVTT_Videos.zip msrvtt_test_1k.json --repo-type dataset --local-dir data
python scripts/prep_frames.py      # 1000 test mp4s -> data/frames/<vid>/{0..7}.jpg, 224x224 (short-side resize + centre crop), segment-centre uniform sampling
python scripts/build_conditions.py # data/manifest.{json,csv}
```
Conditions: `clean`, and `start` (slots 0-3), `middle` (2-5), `end` (4-7) where 4 frames come from 4 other
random 1k-A test videos. Junk seeds are exactly **0,1,2**; per (seed, clip) the *same* 4 junk frames are used at all
three positions, so only position differs. 1000 clips x (1 + 3 pos x 3 seeds) = 10,000 rows.

## Encode
`python scripts/encode.py --model {viclip_b,viclip_l,openclip_meanpool,internvideo2}` -> `emb/<model>/text.npy`,
`video_<cond>_s<seed>.npy`. GPU: `scripts/gpu_job.sh`. **CPU fallback (no GPU): ViCLIP-B** runs on CPU
(`--model viclip_b --device cpu`, ~0.5 s/clip on 8 cores -> ~80 min for 10k rows). InternVideo2 runs with flash-attn
/fused ops off; it needs the InternVideo repo and the gated `OpenGVLab/InternVideo2-CLIP-1B-224p-f8` weights
(no public L14 CLIP checkpoint found) — set `IV2_DIR`, `IV2_CFG`, `IV2_CKPT`. That path is untested.

## Score + pre-registered rule
`python scripts/score.py --emb emb --models internvideo2 viclip_l openclip_meanpool --null openclip_meanpool`
-> `results/block1.csv` (R@1/R@5 per model x condition x seed, mean/std over seeds, per-model PE, worst position,
noise floor) and `results/block1_verdict.json`.
- PE = max - min over {start, middle, end} of seed-mean R@1. Noise floor = 2 x std of R@1 over seeds (max over positions).
- **KILL**: for both video models, PE <= max(1 pt, noise) OR PE <= null PE + 1.
- **GO**: some video model has PE >= null PE + 2 with START worst.
- otherwise **UNCLEAR**.
`python scripts/test_score.py` checks the scorer (random -> KILL, synthetic start-bias -> GO).
Block 2 is out of scope here.

---
## Legacy (earlier smoke attempt, superseded)
## P3 Temporal Embedding Diagnostics

CPU-side preparation for a kill-test on temporal padding with frozen InternVideo2.

### Goal
Smoke-test whether exact text-to-video Recall@1 degrades under three frame conditions (budget=8, k=4):

- **clean**: 8 evenly spaced frames from the clip
- **pad_dup**: 4 kept frames; first 4 slots are duplicates of the kept frames
- **pad_junk**: 4 kept frames; first 4 slots are frames from a different clip

Same kept indices for every condition of a given clip.

### Status
- Frame sampler written (`scripts/frame_sampler.py`)
- Clip source search pending / blocked (no local clips, free short clips with one-line captions not yet materialized into `data/`)
- Manifest not yet written
- Do not train. Frozen InternVideo2 only.

### Usage (GPU machine)
```
cd p3-temporal
git pull
export HF_HOME="$PWD/hf-cache"
export HUGGINGFACE_HUB_CACHE="$PWD/hf-cache/hub"
python scripts/smoke_recall.py   # once manifest + clips exist
```

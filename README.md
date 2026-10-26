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
(`--model viclip_b --device cpu`, ~0.5 s/clip on 8 cores -> ~80 min for 10k rows). InternVideo2 target is the public `OpenGVLab/InternVideo2_CLIP_S`. In the first Block 1 GPU run (Kaggle T4) it **failed to load** on the import `flash_attention_class` -> `flash_attn` (`ModuleNotFoundError: No module named 'flash_attn'`; `results/logs/iv2s.log`). **Fixed:** `encode.py` now builds a patched local copy (`~/.cache/p3/InternVideo2_CLIP_S_noflash`): `FlashAttention` is replaced by a `torch.nn.functional.scaled_dot_product_attention` stub, the naive attention path also uses SDPA, all `flash_attn` imports (FusedMLP, DropoutAddRMSNorm) are removed, and flash-attn / fused RMSNorm / fused MLP are forced off. It is imported as a plain package (no HF dynamic modules). Verified on CPU: loads, 20 clips encode, clean R@1 = 16/20 on that subset. CPU speed ≈ 2.6 s/row (8 cores, fp32) -> ~7 h for 10k rows, so the full run needs the GPU (CUDA: fp32 weights + `torch.autocast("cuda", float16)`; a plain `.half()` broke the fp32 LayerNorms in `mobile_clip_transformer.py` on the T4. Mixed path checked on CPU with `P3_CPU_AUTOCAST=bf16`: cosine to fp32 >= 0.99998). The job pins `transformers<4.50` (requirements.txt), needed by the ViCLIP remote code.

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


## Block 1 results (GPU, Kaggle T4; ViCLIP-L + OpenCLIP at commit 5bb6715, InternVideo2_CLIP_S at 3027f10, fp16 autocast)
Text-to-video on MSR-VTT 1k-A; mean ± std over junk seeds 0,1,2 (clean = single run). Re-scored locally with `scripts/score.py`
(matches GPU Bot's table). Files: `results/block1.csv`, `results/block1_verdict.json`, logs in `results/logs/`.

| model | clean R@1 / R@5 | start R@1 / R@5 | middle R@1 / R@5 | end R@1 / R@5 | PE | worst | noise |
|---|---|---|---|---|---|---|---|
| internvideo2_s | 42.8 / 68.6 | 15.53±0.85 / 35.37±1.04 | 24.90±1.13 / 48.00±0.17 | 22.77±1.53 / 46.27±0.70 | 9.37 | start | 3.06 |
| viclip_l | 39.2 / 63.9 | 18.87±0.15 / 40.07±0.93 | 9.23±0.81 / 25.97±0.55 | 23.73±0.38 / 43.40±1.22 | 14.50 | middle | 1.63 |
| openclip_meanpool (null) | 36.1 / 57.5 | 26.10±0.62 / 48.13±1.18 | 27.03±1.45 / 50.70±0.92 | 26.90±1.21 / 49.97±1.34 | 0.93 | start | 2.89 |

**Verdict: GO** (pre-registered rule): InternVideo2_CLIP_S has PE 9.37 >= null PE 0.93 + 2 with START worst, so KILL does not hold.
(An earlier 2-model run with only ViCLIP-L and the null was UNCLEAR.)

**Caveat: MIDDLE-continuity confound (raised by the P3 Idea Agent).** ViCLIP-L's worst position is MIDDLE, not START. MIDDLE junk
(slots 2-5) is the only condition that splits the 4 real frames into two separate runs (0-1 and 6-7). START and END each keep one
run of 4 real frames in a row. So ViCLIP-L's MIDDLE drop may come from broken temporal continuity, not from a position weighting, and the
three positions differ in continuity as well as in position. This also affects how to read the IV2-S result. Block 2's sliding-evidence
design (with continuity matched across positions) is needed to separate the two. Block 2 has not been started.

## Block 2a (step 1: offset sweep, built, awaiting GPU)
Design: the 4 real frames are always the clip's sampled frames **0,1,2,3 in temporal order** (content fixed, only position moves),
placed as ONE contiguous block at offset k=0..4 of the 8 slots, with the other 4 slots filled by the **same Block 1 junk**
(seeds 0,1,2, same per-clip junk set, same fill order). Plus **SPLIT**: real 0,1 in slots 0-1 and real 2,3 in slots 6-7, junk in 2-5.
Identity check against Block 1 (`scripts/build_block2a.py`): **off0 == Block 1 END for all 3000 rows (reused)**. Block 1 START (real 4-7)
and MIDDLE (real 0,1,6,7) use *different real frames*, so off4 and split are new rows, not reused.
New rows: off1-off4 + split x 3 seeds x 1000 = **15,000** (`data/manifest_block2a.json`). The mean-pool null is order-invariant and sees the
same frame set in every Block 2a condition, so its Block 2a curve equals Block 1 END exactly (flat by construction, D_null = 0); it needs no GPU.
GPU: `tar xzf p3_block2a_bundle.tgz && bash scripts/gpu_job_block2a.sh` (IV2-S + ViCLIP-L; `RUN_NULL=1` also encodes OpenCLIP).
Score: `python scripts/score_block2a.py --emb <dir with Block 1 + Block 2a embeddings per model>` -> `results/block2a.csv`, `results/block2a_verdict.json`.
**Pre-registered read (IV2-S):** EARLY-BIAS (GO) if R@1 decreases monotonically off0->off4 (each step non-increasing within the noise floor, 2x max
seed std) and D = R@1(off0) - R@1(off4) >= D_null + 2. KILL if flat (D <= max(1, noise)) or split explains the gap
(|split - mean(continuous)| >= D). Else UNCLEAR. Also reports split - off2. `scripts/test_score_block2a.py` checks GO / flat-KILL / split-KILL on fake data.

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

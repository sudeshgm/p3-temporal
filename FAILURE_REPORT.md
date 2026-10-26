# P3 failure report: early-frame bias in frozen video-text dual encoders — KILLED

Date: 2026-10-26. Status: **killed by the P3 Idea Agent; no further compute.**

## Hypothesis
Frozen video-text dual encoders give more weight to early frames. So at a fixed 8-frame budget, text-to-video retrieval should
get worse the later the relevant (real) frames sit in the clip. The test had to beat two shortcuts: plain dilution (junk frames remove
evidence) and a mean-pooled image-CLIP null that has no notion of position.

Setup: MSR-VTT 1k-A (1000 clips, one caption each), T=8 evenly spaced frames, 4 real + 4 junk frames taken from other 1k-A test
clips, junk seeds 0,1,2 (the same junk set for a clip across conditions). Models: InternVideo2_CLIP_S (IV2-S), ViCLIP-L,
OpenCLIP ViT-L/14 mean-pooled (the null). Metric: text-to-video R@1/R@5.

## Block 1 — junk at START (slots 0-3) / MIDDLE (2-5) / END (4-7)
R@1 / R@5, mean ± std over seeds (clean = one run). Source: `results/block1.csv`. PE = max − min of R@1 over positions; noise = 2 × max seed std.

| model | clean | start | middle | end | PE | worst | noise |
|---|---|---|---|---|---|---|---|
| internvideo2_s | 42.80 / 68.60 | 15.53±0.85 / 35.37±1.04 | 24.90±1.13 / 48.00±0.17 | 22.77±1.53 / 46.27±0.70 | 9.37 | start | 3.06 |
| viclip_l | 39.20 / 63.90 | 18.87±0.15 / 40.07±0.93 | 9.23±0.81 / 25.97±0.55 | 23.73±0.38 / 43.40±1.22 | 14.50 | middle | 1.63 |
| openclip_meanpool | 36.10 / 57.50 | 26.10±0.62 / 48.13±1.18 | 27.03±1.45 / 50.70±0.92 | 26.90±1.21 / 49.97±1.34 | 0.93 | start | 2.89 |

Pre-registered Block 1 rule: **GO** (IV2-S PE 9.37 ≥ null 0.93 + 2, START worst). Known confound: MIDDLE junk is the only
condition that splits the real frames into two separate runs, so Block 1 mixes up position and continuity.

## Block 2a — one contiguous block of 4 real frames at offset k=0..4, plus SPLIT (real 2 + 2 at both ends)
Source: `results/block2a.csv`. off0 = Block 1 END. The null ignores order, so it is identical in every condition. D = R@1(off0) − R@1(off4).

| model | off0 | off1 | off2 | off3 | off4 | split |
|---|---|---|---|---|---|---|
| internvideo2_s | 22.77±1.53 / 46.27±0.70 | 19.37±1.37 / 40.47±0.95 | 16.47±1.27 / 37.00±0.61 | 16.20±1.73 / 37.03±0.95 | 17.77±1.19 / 38.73±0.47 | 22.77±1.90 / 45.47±0.49 |
| viclip_l | 23.73±0.38 / 43.40±1.22 | 30.07±1.62 / 52.00±0.96 | 31.43±0.84 / 53.80±1.21 | 30.13±0.50 / 52.73±0.45 | 19.57±1.47 / 40.47±1.31 | 8.87±0.81 / 25.43±0.84 |
| openclip_meanpool | 26.90±1.21 / 49.97±1.34 | same | same | same | same | same |

| model | D | noise | split − mean(continuous) | split − off2 | strictly monotonic |
|---|---|---|---|---|---|
| internvideo2_s | 5.00 | 3.80 | +4.25 | +6.30 | no |
| viclip_l | 4.17 | 3.24 | −18.12 | −22.57 | no |
| openclip_meanpool | 0.00 | 2.42 | 0.00 | 0.00 | – (flat) |

## Why it was killed
The stated pre-registered read needed: **offset 0 best, offset 4 worst, a steady decline between them, and split not explaining the gap.**
- **IV2-S fails the monotonic test:** off4 (17.77) > off3 (16.20). The curve bottoms out at off3 and is not a steady early-to-late decline.
- **IV2-S split explains it:** split = off0 = 22.77. Two real frames in slots 0-1 (plus 6-7) do as well as a contiguous block at
  the front. That points to a **first-slot / edge effect**, not a smooth early-frame bias. D = 5.00 is only **1.2 pts above the noise floor** (3.80).
- **ViCLIP-L** is best in the middle (off2 31.43) and collapses on split (8.87). That is a **continuity** effect, not an early-frame
  bias. It also explains its Block 1 MIDDLE drop.
- **Scorer vs the stated read:** `score_block2a.py` printed `EARLY-BIAS (GO)` because it used a *looser* monotonic test
  (each step may rise by up to the noise floor) and the test "split explains if |split − mean(cont)| ≥ D" (4.25 < 5.00, missed by 0.75).
  The read as stated in the plan is stricter (offset 0 best, offset 4 worst, steady decline). Under it, IV2-S fails. The code did not
  encode the rule as written, so the plan's read stands: **KILL**.

## Model substitution
The plan named InternVideo2-CLIP-L14 (8 frames). No public L14 InternVideo2-CLIP checkpoint was found on HF (only the gated 1B/6B
and the public `OpenGVLab/InternVideo2_CLIP_S`), so **IV2-S** was used. It needed two fixes to run on a Kaggle T4 (sm75) and on CPU:
1. A **no-flash patch**: `flash_attention_class.FlashAttention` was swapped for a `torch.nn.functional.scaled_dot_product_attention` stub, the naive
   attention path also uses SDPA, every `flash_attn` import (FusedMLP, DropoutAddRMSNorm) was removed, flash/fused ops were forced off, and the model is loaded as a plain
   package (`scripts/encode.py`, commit 6a22df9).
2. An **fp16 fix**: `.half()` broke the fp32 LayerNorms in `mobile_clip_transformer.py` (Float vs Half). The fix keeps weights fp32 and runs the forward pass
   under `torch.autocast("cuda", float16)`. Checked on CPU with bf16 autocast: cosine to fp32 ≥ 0.99998 (commit 3027f10).
So the result is about a small distilled model, not the planned L14.

## Caveat: real-frame identity differs between blocks
In Block 2a the real frames are always the clip's sampled frames **0-3** (content fixed, only position moves). In Block 1, START
kept real frames **4-7** and MIDDLE kept 0,1,6,7; only END (= off0) is the same condition in both. So the Block 1 START
and MIDDLE numbers are not offsets on the Block 2a curve. They differ in content as well as position.

## What is left over (both unproven)
- A possible **IV2-S first-slot effect**: having real evidence in slot 0 seems to matter (off0 = split ≫ off1-off4), not early frames in general.
- **ViCLIP-L continuity sensitivity**: splitting the real frames into two runs causes a large collapse (−18 pts vs the continuous mean).
Neither was pre-registered as a hypothesis, and neither has been tested on its own.

## Lessons
1. **Put the pre-registered rule in code exactly as written.** Strict monotonicity was in the plan, but the scorer used a looser
   within-noise version and printed GO. Write the scorer from the plan's own wording and unit-test it against edge cases (U-shaped curves, split = endpoint).
2. **Match continuity across conditions from the start.** Block 1's MIDDLE condition mixed position with breaking up the real frames. A contiguous-block
   offset sweep plus an explicit split control should have been Block 1.
3. **Check checkpoint availability before planning.** The named IV2-CLIP-L14 doesn't exist publicly. The swap to IV2-S and its setup cost two GPU runs.
4. **T4 pitfalls:** InternVideo2 remote code hard-imports `flash_attn` even with flash disabled (no sm75 wheels), so patch it out with SDPA.
   Don't call `.half()` on models with custom fp32 LayerNorms; keep weights fp32 and use autocast. Test the mixed path on CPU with bf16 autocast first.

## Compute used
Kaggle T4 (free tier) only: about **2.8 h** (Block 1: ViCLIP-L + OpenCLIP), **0.5 h** (Block 1: IV2-S), **3 h** (Block 2a: IV2-S + ViCLIP-L).
CPU prep and scoring on the shared box. **Nothing paid.**

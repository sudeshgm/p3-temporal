#!/usr/bin/env python3
"""
Smoke test: exact text-to-video Recall@1 / R@5 / MedR
on frozen InternVideo2 (or ViCLIP fallback).
Conditions: clean, pad_dup, pad_junk. Budget=8.
Do not train.
"""
import json
from pathlib import Path
import sys

def main():
    manifest_path = Path("data/manifest.json")
    if not manifest_path.exists():
        print("BLOCKER: data/manifest.json missing. Copy local data/clips and manifest to this machine.")
        sys.exit(1)
    with open(manifest_path) as f:
        manifest = json.load(f)
    n = len(manifest)
    print(f"N_CLIPS: {n}")
    print("Conditions: clean, pad_dup, pad_junk")
    print("Model: Frozen InternVideo2 (smallest stage-2 that fits 16GB T4; fallback to ViCLIP)")
    print("Metrics: exact text-to-video R@1, R@5, MedR")
    print("NOTE: Skeleton. Implement frame sampling via scripts/frame_sampler.py")
    print("      and embedding extraction. Clips are local only (not in repo).")
    results = {"n_clips": n, "conditions": ["clean", "pad_dup", "pad_junk"], "status": "skeleton"}
    Path("results").mkdir(exist_ok=True)
    with open("results/smoke_placeholder.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Wrote results/smoke_placeholder.json")

if __name__ == "__main__":
    main()

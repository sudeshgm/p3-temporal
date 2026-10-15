#!/usr/bin/env python3
"""
16-clip smoke test for text-to-video Recall@1.
Conditions: clean, pad_dup, pad_junk.
Frozen InternVideo2.
Do not train.
"""
import json
from pathlib import Path

def main():
    manifest_path = Path("data/manifest.json")
    if not manifest_path.exists():
        print("BLOCKER: data/manifest.json missing")
        return
    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f"N_CLIPS: {len(manifest)}")
    print("GPU skeleton only — implement frozen InternVideo2 embeddings + exact R@1 once clips present")

if __name__ == "__main__":
    main()

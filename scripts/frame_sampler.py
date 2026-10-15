#!/usr/bin/env python3
"""
Frame sampler for P3 temporal diagnostics.
Budget=8, k=4.

Conditions:
- clean: 8 evenly spaced frames from the clip.
- pad_dup: 4 kept frames; first 4 slots are duplicates of the kept frames.
- pad_junk: 4 kept frames; first 4 slots are frames from a different (junk) clip.
Same kept indices for every condition of a given clip.
"""

import numpy as np
from typing import List, Dict, Optional


def get_even_indices(total_frames: int, num: int) -> List[int]:
    """Return num evenly spaced indices in [0, total_frames-1]."""
    if total_frames < num:
        raise ValueError(f"Need at least {num} frames, got {total_frames}")
    return np.linspace(0, total_frames - 1, num, dtype=int).tolist()


def sample_frames(total_frames: int, junk_total_frames: Optional[int] = None) -> Dict[str, List[int]]:
    """
    Returns dict of condition -> list of 8 frame indices.
    For pad_junk the first 4 are junk indices, last 4 are kept from main clip.
    Kept indices are the same 4 across conditions.
    """
    if total_frames < 8:
        raise ValueError("Clip must have at least 8 frames")

    clean_indices = get_even_indices(total_frames, 8)
    kept_indices = clean_indices[::2]  # 4 indices, same for all conditions

    pad_dup_indices = kept_indices + kept_indices

    if junk_total_frames is None or junk_total_frames < 4:
        junk_indices = [0, 0, 0, 0]
    else:
        junk_indices = get_even_indices(junk_total_frames, 4)
    pad_junk_indices = junk_indices + kept_indices

    return {
        "clean": clean_indices,
        "pad_dup": pad_dup_indices,
        "pad_junk": pad_junk_indices,
        "kept": kept_indices,
    }


if __name__ == "__main__":
    print(sample_frames(100, 50))

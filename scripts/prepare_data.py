#!/usr/bin/env python3
"""
Recreate data/clips and data/manifest.json on the GPU machine.
Downloads public sample MP4s and extracts short segments.
No login required. Continues on individual failures.
"""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLIPS = ROOT / "data" / "clips"
CLIPS.mkdir(parents=True, exist_ok=True)

DOWNLOADS = [
    ("clip_01.mp4", "https://interactive-examples.mdn.mozilla.net/media/cc0-videos/flower.mp4"),
    ("clip_02.mp4", "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4"),
    ("clip_03.mp4", "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/Big_Buck_Bunny_720_10s_5MB.mp4"),
    ("clip_05.mp4", "https://filesamples.com/samples/video/mp4/sample_640x360.mp4"),
    ("clip_06.mp4", "https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-mp4-file.mp4"),
    ("clip_11.mp4", "https://cdn.truefilesize.com/mp4/sample-1mb.mp4"),
    ("clip_12.mp4", "https://cdn.truefilesize.com/mp4/sample-5mb.mp4"),
]

def download(name, url):
    dest = CLIPS / name
    if dest.exists() and dest.stat().st_size > 10000:
        print(f"exists {name}")
        return True
    print(f"downloading {name}")
    r = subprocess.run(
        ["curl", "-L", "-f", "-o", str(dest), url, "--max-time", "90", "-A", "Mozilla/5.0"],
        capture_output=True,
    )
    ok = r.returncode == 0 and dest.exists() and dest.stat().st_size > 1000
    print(f"{'OK' if ok else 'FAIL'} {name}")
    return ok

def extract(src_name, dest_name, ss, dur):
    src = CLIPS / src_name
    dest = CLIPS / dest_name
    if not src.exists():
        print(f"skip extract {dest_name}, missing {src_name}")
        return False
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"exists {dest_name}")
        return True
    print(f"extracting {dest_name}")
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ss", str(ss), "-t", str(dur),
         "-c:v", "libx264", "-preset", "ultrafast", "-an", str(dest)],
        capture_output=True,
    )
    ok = r.returncode == 0 and dest.exists() and dest.stat().st_size > 1000
    print(f"{'OK' if ok else 'FAIL'} {dest_name}")
    return ok

# Downloads
for name, url in DOWNLOADS:
    download(name, url)

# Extracts from longer clips
extract("clip_05.mp4", "clip_13.mp4", 1, 4)
extract("clip_05.mp4", "clip_14b.mp4", 6, 4)
extract("clip_06.mp4", "clip_15b.mp4", 10, 5)
extract("clip_06.mp4", "clip_16b.mp4", 40, 5)
extract("clip_06.mp4", "clip_17b.mp4", 80, 5)
extract("clip_02.mp4", "clip_18.mp4", 0, 4)
extract("clip_03.mp4", "clip_19.mp4", 2, 4)
extract("clip_05.mp4", "clip_14.mp4", 5, 5)
extract("clip_06.mp4", "clip_15.mp4", 0, 5)

# Build manifest matching previous captions
captions = {
    "clip_01.mp4": "a flower blooming in close-up",
    "clip_02.mp4": "animated rabbit running through a forest",
    "clip_03.mp4": "cartoon rabbit exploring green woods",
    "clip_05.mp4": "person walking outdoors in daylight",
    "clip_06.mp4": "scenic nature landscape with movement",
    "clip_11.mp4": "abstract color test pattern moving",
    "clip_12.mp4": "high definition sample video of shapes",
    "clip_13.mp4": "short outdoor scene with motion",
    "clip_14b.mp4": "continued outdoor footage segment",
    "clip_15b.mp4": "nature clip showing trees and sky",
    "clip_16b.mp4": "another short nature sequence",
    "clip_17b.mp4": "landscape view with camera movement",
    "clip_18.mp4": "animated character in colorful environment",
    "clip_19.mp4": "cartoon animal moving through grass",
    "clip_14.mp4": "brief outdoor activity clip",
    "clip_15.mp4": "short scenic video segment",
}

order = list(captions.keys())
manifest = []
for i, fname in enumerate(order):
    p = CLIPS / fname
    if not p.exists() or p.stat().st_size < 1000:
        continue
    junk = order[(i + 1) % len(order)]
    manifest.append({
        "clip_path": f"data/clips/{fname}",
        "caption": captions[fname],
        "junk_clip_path": f"data/clips/{junk}",
    })

man_path = ROOT / "data" / "manifest.json"
with open(man_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"Wrote manifest with {len(manifest)} entries to {man_path}")

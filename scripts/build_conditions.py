"""Manifest: clip x condition x seed -> 8 frame paths. Junk = 4 frames from other test videos (seeds 0,1,2)."""
import json, os, random, csv
D = os.environ.get("P3_DATA", "/workspace/p3/data")
POS = {"start": [0,1,2,3], "middle": [2,3,4,5], "end": [4,5,6,7]}; SEEDS = [0,1,2]
test = json.load(open(f"{D}/msrvtt_test_1k.json"))
vid_of = lambda e: e.get("video_id") or e["video"].split(".")[0]
vids = [vid_of(e) for e in test]
caps = {vid_of(e): (e.get("caption") if isinstance(e.get("caption"), str) else (e.get("caption") or e.get("sentence"))) for e in test}
fp = lambda v, k: f"frames/{v}/{k}.jpg"
rows = [dict(clip=v, condition="clean", seed=-1, frames=[fp(v,k) for k in range(8)], junk_src=[]) for v in vids]
for s in SEEDS:
    for i, v in enumerate(vids):
        rng = random.Random(s*100003 + i); others = [u for u in vids if u != v]
        src = rng.sample(others, 4); jf = [(u, rng.randrange(8)) for u in src]  # same junk set at every position
        for pos, slots in POS.items():
            fr = [fp(v,k) for k in range(8)]
            for slot, (u, k) in zip(slots, jf): fr[slot] = fp(u, k)
            rows.append(dict(clip=v, condition=pos, seed=s, frames=fr, junk_src=[f"{u}/{k}" for u,k in jf]))
json.dump(dict(root=D, clips=vids, captions=caps, rows=rows), open(f"{D}/manifest.json","w"))
with open(f"{D}/manifest.csv","w",newline="") as f:
    w = csv.writer(f); w.writerow(["clip","condition","seed"]+[f"f{k}" for k in range(8)])
    for r in rows: w.writerow([r["clip"], r["condition"], r["seed"]]+r["frames"])
print("clips", len(vids), "rows", len(rows))

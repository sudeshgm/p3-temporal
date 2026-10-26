"""Block 2a manifest: ONE contiguous block of 4 real frames at offset k=0..4 in 8 slots, + SPLIT (2 real | 4 junk | 2 real).
Real frames are ALWAYS the clip's sampled frames 0,1,2,3 in temporal order (content fixed; only position moves).
Junk = Block 1 junk set (same seed 0,1,2 and same per-clip 4 junk frames, filled into junk slots in the same order).
Identity with Block 1 (checked below):  off0 == Block1 END (real 0-3 in slots 0-3, junk jf[0..3] in 4-7) -> reused.
  Block1 START = real 4-7 at slots 4-7 and Block1 MIDDLE = real 0,1,6,7: different real frames, so off4 and split are NEW.
New rows: off1..off4 + split = 5 conds x 3 seeds x 1000 = 15,000. Written to data/manifest_block2a.json (+ .csv)."""
import json, os, csv
D = os.environ.get("P3_DATA", "/workspace/p3/data")
m1 = json.load(open(f"{D}/manifest.json")); vids = m1["clips"]
fp = lambda v, k: f"frames/{v}/{k}.jpg"
junk = {(r["clip"], r["seed"]): r["junk_src"] for r in m1["rows"] if r["condition"] == "start"}
b1 = {(r["clip"], r["condition"], r["seed"]): r["frames"] for r in m1["rows"]}
def layout(v, s, real_slots):
    jf = [fp(*j.split("/")) for j in junk[(v, s)]]; real = [fp(v, k) for k in range(4)]
    fr, ri, ji = [None]*8, iter(real), iter(jf)
    for slot in range(8): fr[slot] = next(ri) if slot in real_slots else next(ji)
    return fr
CONDS = {f"off{k}": set(range(k, k+4)) for k in range(5)}; CONDS["split"] = {0, 1, 6, 7}
rows, reused, n_same = [], 0, {}
for s in (0, 1, 2):
    for v in vids:
        for c, slots in CONDS.items():
            fr = layout(v, s, slots)
            for c1 in ("start", "middle", "end"):
                if fr == b1[(v, c1, s)]: n_same[(c, c1)] = n_same.get((c, c1), 0) + 1
            if c == "off0":
                assert fr == b1[(v, "end", s)]; reused += 1; continue
            rows.append(dict(clip=v, condition=f"b2a_{c}", seed=s, frames=fr, junk_src=junk[(v, s)]))
print("identity with Block 1 (cond, block1cond): rows identical =", n_same)
json.dump(dict(root=D, clips=vids, captions=m1["captions"], rows=rows, reuse={"b2a_off0": "end"}),
          open(f"{D}/manifest_block2a.json", "w"))
with open(f"{D}/manifest_block2a.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["clip","condition","seed"]+[f"f{k}" for k in range(8)])
    for r in rows: w.writerow([r["clip"], r["condition"], r["seed"]]+r["frames"])
print("new rows", len(rows), "reused (off0 = Block1 end)", reused, "unique frames", len({f for r in rows for f in r["frames"]}))

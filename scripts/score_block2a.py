"""Block 2a scoring. Per model dir EMB/<model>: text.npy, video_end_s{s}.npy (Block 1, = off0),
video_b2a_off{1..4}_s{s}.npy, video_b2a_split_s{s}.npy. Order-invariant models (--orderless, default openclip_meanpool)
see the SAME frame set in every Block 2a condition, so all their conditions = Block 1 END (no GPU needed).
Outputs results/block2a.csv and results/block2a_verdict.json.
Pre-registered read (on --target, default internvideo2_s):
  EARLY-BIAS (GO) iff target R@1 decreases monotonically off0->off4 AND D >= D_null + 2, D = R@1(off0) - R@1(off4).
  KILL iff curve is flat (D <= max(1, noise)) OR split explains the gap (split-vs-continuous gap magnitude >= D).
  else UNCLEAR. Monotonic = each step off_k -> off_k+1 is non-increasing within the step noise (2 x max seed std)."""
import argparse, os, json, csv, numpy as np
from score import recall
SEEDS = [0, 1, 2]; OFF = [f"off{k}" for k in range(5)]; CONDS = OFF + ["split"]
def vpath(d, c, s, orderless):
    if c == "off0" or orderless: return f"{d}/video_end_s{s}.npy"
    return f"{d}/video_b2a_{c}_s{s}.npy"
def score_model(d, orderless):
    t = np.load(f"{d}/text.npy")
    return {(c, s): recall(t, np.load(vpath(d, c, s, orderless))) for c in CONDS for s in SEEDS}
def summarize(r):
    mean = {c: {k: float(np.mean([r[(c, s)][k] for s in SEEDS])) for k in (1, 5)} for c in CONDS}
    std = {c: {k: float(np.std([r[(c, s)][k] for s in SEEDS], ddof=1)) for k in (1, 5)} for c in CONDS}
    m1 = [mean[c][1] for c in OFF]; noise = 2 * max(std[c][1] for c in CONDS)
    D = m1[0] - m1[4]; cont = float(np.mean(m1))
    mono_strict = all(m1[i+1] <= m1[i] for i in range(4))
    mono_tol = all(m1[i+1] <= m1[i] + noise for i in range(4)) and D > 0
    return dict(R1_by_offset=m1, D=D, noise=noise, split_R1=mean["split"][1],
                split_minus_cont_mean=mean["split"][1] - cont, split_minus_off2=mean["split"][1] - m1[2],
                monotonic_strict=mono_strict, monotonic_within_noise=mono_tol, mean=mean, std=std)
def verdict(S, target, null):
    t, n = S[target], S[null]
    flat = t["D"] <= max(1.0, t["noise"])
    split_explains = abs(t["split_minus_cont_mean"]) >= abs(t["D"])
    go = t["monotonic_within_noise"] and t["D"] >= n["D"] + 2
    v = "KILL" if (flat or split_explains) else ("EARLY-BIAS (GO)" if go else "UNCLEAR")
    return v, dict(flat=flat, split_explains=split_explains, go_condition=go)
def main(a):
    res = {m: score_model(f"{a.emb}/{m}", m in a.orderless) for m in a.models}
    S = {m: summarize(r) for m, r in res.items()}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "condition", "seed", "R@1", "R@5"])
        for m, r in res.items():
            for (c, s), v in r.items(): w.writerow([m, c, s, f"{v[1]:.2f}", f"{v[5]:.2f}"])
            for c in CONDS: w.writerow([m, c, "mean±std", f"{S[m]['mean'][c][1]:.2f}±{S[m]['std'][c][1]:.2f}",
                                        f"{S[m]['mean'][c][5]:.2f}±{S[m]['std'][c][5]:.2f}"])
            x = S[m]; w.writerow([m, "summary", f"D={x['D']:.2f}", f"noise={x['noise']:.2f}",
                f"split-cont={x['split_minus_cont_mean']:.2f};split-off2={x['split_minus_off2']:.2f};mono={x['monotonic_within_noise']}"])
    v, why = verdict(S, a.target, a.null)
    json.dump(dict(verdict=v, target=a.target, null=a.null, checks=why, orderless=a.orderless, summary=S),
              open(a.out.replace(".csv", "_verdict.json"), "w"), indent=1)
    for m in a.models:
        x = S[m]; print(m, "R1 by offset", [round(y, 2) for y in x["R1_by_offset"]], "split", round(x["split_R1"], 2),
                        "D", round(x["D"], 2), "noise", round(x["noise"], 2), "mono", x["monotonic_within_noise"])
    print("VERDICT:", v, why)
if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--emb", required=True)
    p.add_argument("--models", nargs="+", default=["internvideo2_s", "viclip_l", "openclip_meanpool"])
    p.add_argument("--target", default="internvideo2_s"); p.add_argument("--null", default="openclip_meanpool")
    p.add_argument("--orderless", nargs="*", default=["openclip_meanpool"])
    p.add_argument("--out", default="/workspace/p3/results/block2a.csv")
    main(p.parse_args())

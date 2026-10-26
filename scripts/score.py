"""Score Block 1. Input dir per model: text.npy [N,d] (row i = caption of clip i, manifest clip order) and
video_<condition>_s<seed>.npy [N,d] (clean: video_clean_s-1.npy). Writes results/block1.csv and verdict.json.
Usage: score.py --emb EMB_ROOT --models internvideo2 viclip_l openclip_meanpool --null openclip_meanpool"""
import argparse, os, json, csv, numpy as np
POS = ["start","middle","end"]; SEEDS = [0,1,2]
def recall(t, v, ks=(1,5)):
    t = t/np.linalg.norm(t,axis=1,keepdims=True); v = v/np.linalg.norm(v,axis=1,keepdims=True)
    S = t @ v.T; rank = (S > np.diag(S)[:,None]).sum(1)
    return {k: 100*float((rank < k).mean()) for k in ks}
def score_model(d):
    t = np.load(f"{d}/text.npy"); out = {("clean",-1): recall(t, np.load(f"{d}/video_clean_s-1.npy"))}
    for p in POS:
        for s in SEEDS: out[(p,s)] = recall(t, np.load(f"{d}/video_{p}_s{s}.npy"))
    return out
def summarize(r):
    m = {p: float(np.mean([r[(p,s)][1] for s in SEEDS])) for p in POS}
    sd = {p: float(np.std([r[(p,s)][1] for s in SEEDS], ddof=1)) for p in POS}
    pe = max(m.values()) - min(m.values())
    return dict(PE=pe, worst=min(m, key=m.get), noise=2*max(sd.values()), mean_r1=m, std_r1=sd)
def verdict(summ, video_models, null):
    npe = summ[null]["PE"]
    kill = all(summ[v]["PE"] <= max(1.0, summ[v]["noise"]) or summ[v]["PE"] <= npe + 1 for v in video_models)
    go = any(summ[v]["PE"] >= npe + 2 and summ[v]["worst"] == "start" for v in video_models)
    return "KILL" if kill else ("GO" if go else "UNCLEAR")  # KILL takes precedence (they cannot both hold for one model)
def main(a):
    res = {m: score_model(f"{a.emb}/{m}") for m in a.models}
    summ = {m: summarize(r) for m, r in res.items()}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model","condition","seed","R@1","R@5"])
        for m, r in res.items():
            for (c, s), v in r.items(): w.writerow([m, c, s, f"{v[1]:.2f}", f"{v[5]:.2f}"])
            for c in POS:
                for k in (1,5):
                    xs = [r[(c,s)][k] for s in SEEDS]
                    w.writerow([m, c, f"mean_R@{k}", f"{np.mean(xs):.2f}", ""]); w.writerow([m, c, f"std_R@{k}", f"{np.std(xs,ddof=1):.2f}", ""])
            S = summ[m]; w.writerow([m, "summary", f"PE={S['PE']:.2f}", f"worst={S['worst']}", f"noise={S['noise']:.2f}"])
    vids = [m for m in a.models if m != a.null]; v = verdict(summ, vids, a.null)
    json.dump(dict(verdict=v, null=a.null, summary=summ), open(a.out.replace(".csv","_verdict.json"),"w"), indent=1)
    for m in a.models: print(m, {k: (round(x,2) if isinstance(x,float) else x) for k,x in summ[m].items() if k in ("PE","worst","noise")})
    print("VERDICT:", v)
if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--emb", required=True)
    p.add_argument("--models", nargs="+", default=["internvideo2","viclip_l","openclip_meanpool"])
    p.add_argument("--null", default="openclip_meanpool"); p.add_argument("--out", default="/workspace/p3/results/block1.csv")
    main(p.parse_args())

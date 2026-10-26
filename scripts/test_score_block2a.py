"""Fake-data checks: (a) monotonic early bias -> GO, (b) flat -> KILL, (c) split dip explains gap -> KILL."""
import numpy as np, os, subprocess, sys, tempfile, json
here = os.path.dirname(os.path.abspath(__file__)); rng = np.random.default_rng(0); N, d = 1000, 64
def mk(dirn, noise_by_cond, t):
    os.makedirs(dirn, exist_ok=True); np.save(f"{dirn}/text.npy", t)
    for c, a in noise_by_cond.items():
        for s in range(3):
            name = "video_end" if c == "off0" else f"video_b2a_{c}"
            np.save(f"{dirn}/{name}_s{s}.npy", t + a * rng.standard_normal((N, d)))
cases = {"early_bias": ({f"off{k}": 3.0 + 0.6*k for k in range(5)} | {"split": 4.2}, "EARLY-BIAS (GO)"),
         "flat": ({f"off{k}": 4.0 for k in range(5)} | {"split": 4.0}, "KILL"),
         "split_explains": ({"off0": 3.6, "off1": 3.7, "off2": 3.8, "off3": 3.9, "off4": 4.0, "split": 6.0}, "KILL")}
ok = True
for name, (cfg, want) in cases.items():
    r = tempfile.mkdtemp(); t = rng.standard_normal((N, d))
    mk(f"{r}/internvideo2_s", cfg, t); mk(f"{r}/viclip_l", {c: 4.0 for c in cfg}, t)
    mk(f"{r}/openclip_meanpool", {"off0": 4.0}, t)  # orderless: only Block 1 END needed
    out = subprocess.run([sys.executable, f"{here}/score_block2a.py", "--emb", r, "--out", f"{r}/b2a.csv"], capture_output=True, text=True, check=True).stdout
    got = json.load(open(f"{r}/b2a_verdict.json"))["verdict"]; print(f"== {name}: got {got}, want {want}"); print(out.strip().splitlines()[0]); ok &= got == want
print("ALL PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)

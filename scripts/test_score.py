"""Self-test: random embeddings (expect KILL) and a synthetic START-bias model (expect GO)."""
import numpy as np, os, subprocess, sys, tempfile
def mk(d, N=1000, dim=64, bias=None, rng=None):
    os.makedirs(d, exist_ok=True); t = rng.standard_normal((N,dim)); np.save(f"{d}/text.npy", t)
    noise = lambda a: t + a*rng.standard_normal((N,dim))
    np.save(f"{d}/video_clean_s-1.npy", noise(3.0) if bias else rng.standard_normal((N,dim)))
    for p in ["start","middle","end"]:
        for s in range(3):
            a = {"start":6.0,"middle":4.5,"end":4.0}[p] if bias else None
            np.save(f"{d}/video_{p}_s{s}.npy", noise(a) if bias else rng.standard_normal((N,dim)))
here = os.path.dirname(os.path.abspath(__file__)); rng = np.random.default_rng(0)
for name, b in [("random", False), ("startbias", True)]:
    r = tempfile.mkdtemp()
    mk(f"{r}/internvideo2", bias=b, rng=rng); mk(f"{r}/viclip_l", bias=False, rng=rng); mk(f"{r}/openclip_meanpool", bias=False, rng=rng)
    print("==", name); subprocess.run([sys.executable, f"{here}/score.py", "--emb", r, "--out", f"{r}/block1.csv"], check=True)

"""Extract 1k-A test videos, decode T=8 uniform frames (224x224 short-side resize + center crop, JPG)."""
import json, zipfile, os, sys, cv2, numpy as np
from concurrent.futures import ProcessPoolExecutor
D = os.environ.get("P3_DATA", "/workspace/p3/data"); T = 8
def load_test():
    j = json.load(open(f"{D}/msrvtt_test_1k.json"))
    return j
def vid_of(e): return e.get("video_id") or e.get("video") .split(".")[0]
def decode(vid):
    out = f"{D}/frames/{vid}"
    if os.path.exists(f"{out}/7.jpg"): return vid, True
    os.makedirs(out, exist_ok=True)
    cap = cv2.VideoCapture(f"{D}/videos/{vid}.mp4"); frames = []
    while True:
        ok, f = cap.read()
        if not ok: break
        frames.append(f)
    if len(frames) == 0: return vid, False
    idx = np.linspace(0, len(frames)-1, 2*T+1)[1::2].round().astype(int)  # segment centres
    for k, i in enumerate(idx):
        f = frames[i]; h, w = f.shape[:2]; s = 224/min(h, w)
        f = cv2.resize(f, (round(w*s), round(h*s)), interpolation=cv2.INTER_CUBIC)
        h, w = f.shape[:2]; y, x = (h-224)//2, (w-224)//2
        cv2.imwrite(f"{out}/{k}.jpg", f[y:y+224, x:x+224], [cv2.IMWRITE_JPEG_QUALITY, 95])
    return vid, True
if __name__ == "__main__":
    test = load_test(); vids = sorted({vid_of(e) for e in test})
    print("test entries", len(test), "videos", len(vids))
    os.makedirs(f"{D}/videos", exist_ok=True)
    with zipfile.ZipFile(f"{D}/MSRVTT_Videos.zip") as z:
        names = {os.path.basename(n).split(".")[0]: n for n in z.namelist() if n.endswith(".mp4")}
        for v in vids:
            p = f"{D}/videos/{v}.mp4"
            if not os.path.exists(p): open(p, "wb").write(z.read(names[v]))
    with ProcessPoolExecutor(8) as ex: res = list(ex.map(decode, vids))
    bad = [v for v, ok in res if not ok]; print("decoded", len(res)-len(bad), "failed", bad)

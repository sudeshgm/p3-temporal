"""Block 1 encoder forward passes. Reads data/manifest.json, writes EMB/<model>/text.npy and
video_<cond>_s<seed>.npy (rows in manifest clip order). Unique frame-tuples are encoded once.
Models: viclip_b (CPU fallback), viclip_l, openclip_meanpool (ViT-L-14 openai, mean of 8 normalised frame
embeddings = no-position null), internvideo2 (InternVideo2-CLIP, 8 frames, flash-attn OFF; see README).
Usage: python encode.py --model viclip_l [--device cuda] [--limit N] [--bs 16]"""
import argparse, json, os, sys, numpy as np, torch, cv2
from huggingface_hub import snapshot_download
MEAN_IN = np.array([0.485,0.456,0.406]); STD_IN = np.array([0.229,0.224,0.225])
MEAN_CLIP = np.array([0.48145466,0.4578275,0.40821073]); STD_CLIP = np.array([0.26862954,0.26130258,0.27577711])
def load_clip(root, frames, mean, std):  # -> [8,3,224,224]
    x = np.stack([cv2.imread(f"{root}/{f}")[:, :, ::-1] for f in frames]).astype(np.float32)/255.
    return torch.from_numpy(((x-mean)/std).transpose(0,3,1,2)).float()

class ViCLIP:
    def __init__(s, size, dev):
        from transformers import AutoModel
        repo = snapshot_download(f"OpenGVLab/ViCLIP-{'B-16' if size=='b' else 'L-14'}-hf")
        cfgp = f"{repo}/config.json"; c = json.load(open(cfgp))
        if not os.path.isabs(c["tokenizer_path"]):  # repo ships a relative tokenizer path
            c["tokenizer_path"] = f"{repo}/bpe_simple_vocab_16e6.txt.gz"
            if os.path.islink(cfgp): os.remove(cfgp)
            json.dump(c, open(cfgp, "w"))
        s.m = AutoModel.from_pretrained(repo, trust_remote_code=True).to(dev).eval(); s.dev = dev; s.mean, s.std = MEAN_IN, STD_IN
    def video(s, x): return s.m.get_vid_features(x.to(s.dev)).float()   # x [B,8,3,H,W]
    def text(s, caps): return torch.cat([s.m.get_text_features(c, s.m.tokenizer, {}) for c in caps]).float()

class OpenCLIPMean:
    def __init__(s, dev):
        import open_clip
        s.m, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai"); s.m = s.m.to(dev).eval()
        s.tok = open_clip.get_tokenizer("ViT-L-14"); s.dev = dev; s.mean, s.std = MEAN_CLIP, STD_CLIP
    def video(s, x):
        B, T = x.shape[:2]; f = s.m.encode_image(x.flatten(0,1).to(s.dev)).float()
        return (f/f.norm(dim=-1, keepdim=True)).view(B, T, -1).mean(1)
    def text(s, caps): return s.m.encode_text(s.tok(caps).to(s.dev)).float()

class InternVideo2:
    """UNTESTED (needs GPU + gated weights). Setup: git clone https://github.com/OpenGVLab/InternVideo;
    IV2_DIR=InternVideo/InternVideo2/multi_modality; IV2_CFG=its CLIP config (8 frames, 224);
    IV2_CKPT=1B_clip.pth from OpenGVLab/InternVideo2-CLIP-1B-224p-f8 (gated: accept terms, set HF_TOKEN) + towers per its README.
    No public 'L14' InternVideo2-CLIP checkpoint found on HF; 1B-224p-f8 is the closest public 8-frame one."""
    def __init__(s, dev):
        sys.path.insert(0, os.environ["IV2_DIR"])
        from utils.config import Config; from models import InternVideo2_CLIP
        cfg = Config.from_file(os.environ["IV2_CFG"])
        for k in ("use_flash_attn", "use_fused_rmsnorm", "use_fused_mlp"): cfg.model.vision_encoder[k] = False  # flash-attn OFF
        if "use_flash_attn" in cfg.model.text_encoder: cfg.model.text_encoder.use_flash_attn = False
        cfg.model.vision_encoder.num_frames = 8
        s.m = InternVideo2_CLIP(config=cfg, is_pretrain=False)
        sd = torch.load(os.environ["IV2_CKPT"], map_location="cpu"); print(s.m.load_state_dict(sd.get("module", sd), strict=False))
        s.m = s.m.to(dev).eval(); s.m = s.m.half() if dev == "cuda" else s.m; s.dev = dev; s.mean, s.std = MEAN_IN, STD_IN
    def video(s, x): return s.m.encode_vision(x.to(s.dev, dtype=next(s.m.parameters()).dtype), test=True).float()
    def text(s, caps):
        t = s.m.tokenizer(caps, padding="max_length", truncation=True, max_length=40, return_tensors="pt").to(s.dev)
        return s.m.encode_text(t).float()

def main(a):
    man = json.load(open(a.manifest)); root = man["root"]; clips = man["clips"][:a.limit] if a.limit else man["clips"]
    keep = set(clips); rows = [r for r in man["rows"] if r["clip"] in keep]
    M = {"viclip_b": lambda: ViCLIP("b", a.device), "viclip_l": lambda: ViCLIP("l", a.device),
         "openclip_meanpool": lambda: OpenCLIPMean(a.device), "internvideo2": lambda: InternVideo2(a.device)}[a.model]()
    out = f"{a.emb}/{a.model}"; os.makedirs(out, exist_ok=True)
    with torch.no_grad():
        caps = [man["captions"][c] for c in clips]
        np.save(f"{out}/text.npy", torch.cat([M.text(caps[i:i+64]) for i in range(0, len(caps), 64)]).cpu().numpy())
        uniq = dict.fromkeys(tuple(r["frames"]) for r in rows); keys = list(uniq)
        print(f"{a.model}: {len(clips)} clips, {len(rows)} rows, {len(keys)} unique frame tuples", flush=True)
        for i in range(0, len(keys), a.bs):
            x = torch.stack([load_clip(root, k, M.mean, M.std) for k in keys[i:i+a.bs]])
            for k, e in zip(keys[i:i+a.bs], M.video(x).cpu().numpy()): uniq[k] = e
            if (i//a.bs) % 20 == 0: print(f"  {i+len(x)}/{len(keys)}", flush=True)
    idx = {c: j for j, c in enumerate(clips)}; groups = {}
    for r in rows: groups.setdefault((r["condition"], r["seed"]), [None]*len(clips))[idx[r["clip"]]] = uniq[tuple(r["frames"])]
    for (c, s), v in groups.items(): np.save(f"{out}/video_{c}_s{s}.npy", np.stack(v))
    print("wrote", out, sorted(groups))
if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--model", required=True)
    p.add_argument("--manifest", default="/workspace/p3/data/manifest.json"); p.add_argument("--emb", default="/workspace/p3/emb")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=0); p.add_argument("--bs", type=int, default=16)
    main(p.parse_args())

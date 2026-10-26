"""Block 1 encoder forward passes. Reads data/manifest.json, writes EMB/<model>/text.npy and
video_<cond>_s<seed>.npy (rows in manifest clip order). Unique frame-tuples are encoded once.
Models: viclip_b (CPU fallback), viclip_l, openclip_meanpool (ViT-L-14 openai, mean of 8 normalised frame
embeddings = no-position null), internvideo2 (public InternVideo2_CLIP_S, 8 frames, flash_attn-free SDPA patch).
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

IV2S_FLASH_STUB = '''"""P3 patch: flash_attn-free FlashAttention using torch SDPA (works on CPU and sm75/T4)."""
import torch, torch.nn as nn, torch.nn.functional as F
class FlashAttention(nn.Module):
    def __init__(self, softmax_scale=None, attention_dropout=0.0, device=None, dtype=None):
        super().__init__(); self.softmax_scale = softmax_scale; self.dropout_p = attention_dropout
    def forward(self, qkv, key_padding_mask=None, causal=False, cu_seqlens=None, max_s=None, need_weights=False):
        q, k, v = qkv.unbind(2)  # (B,S,3,H,D) -> (B,S,H,D)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        m = None if key_padding_mask is None else key_padding_mask[:, None, None, :].bool()
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=m, dropout_p=self.dropout_p if self.training else 0.0,
                                           is_causal=causal and m is None, scale=self.softmax_scale)
        return o.transpose(1, 2), None
'''
SDPA_NAIVE = '''        attn = ((q * self.scale) @ k.transpose(-2, -1))
        # attn = attn - attn.max(-1)[0].unsqueeze(-1)  # in case of overflow for fp16
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)'''
SDPA_NEW = '''        x = F.scaled_dot_product_attention(q, k, v, scale=self.scale).transpose(1, 2).reshape(B, N, C)  # P3 patch: SDPA'''

def patched_iv2s_repo():
    """Copy OpenGVLab/InternVideo2_CLIP_S remote code to a local dir with all flash_attn imports removed and
    fused ops / flash-attn forced off in config.json."""
    import shutil
    src = snapshot_download("OpenGVLab/InternVideo2_CLIP_S")
    dst = os.path.expanduser("~/.cache/p3/InternVideo2_CLIP_S_noflash")
    if not os.path.exists(f"{dst}/.patched"):
        shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst, symlinks=False)
        open(f"{dst}/flash_attention_class.py", "w").write(IV2S_FLASH_STUB)
        for fn in ("internvideo2_clip_vision.py", "internvideo2.py"):
            t = open(f"{dst}/{fn}").read()
            t = t.replace("from flash_attn.modules.mlp import FusedMLP", "FusedMLP = None  # P3 patch: no flash_attn")
            t = t.replace("from flash_attn.modules.mlp import Mlp as FusedMLP", "FusedMLP = None  # P3 patch")
            t = t.replace("from flash_attn.ops.rms_norm import DropoutAddRMSNorm", "DropoutAddRMSNorm = None  # P3 patch")
            assert SDPA_NAIVE in t or fn == "internvideo2.py", fn
            t = t.replace(SDPA_NAIVE, SDPA_NEW)
            assert "import flash_attn" not in t and "from flash_attn" not in t, fn
            open(f"{dst}/{fn}", "w").write(t)
        c = json.load(open(f"{dst}/config.json")); ve = c["model"]["vision_encoder"]
        ve.update(use_flash_attn=False, use_fused_rmsnorm=False, use_fused_mlp=False); c["device"] = "cpu"
        c["torch_dtype"] = "float32"; json.dump(c, open(f"{dst}/config.json", "w"), indent=1)
        open(f"{dst}/.patched", "w").write("ok")
    return dst

class InternVideo2:
    """Public OpenGVLab/InternVideo2_CLIP_S (8 frames, 224), flash_attn-free patched copy (SDPA attention, no fused ops).
    fp32 on CPU; fp16 on CUDA (T4/sm75 fine: SDPA math/mem-efficient kernels)."""
    def __init__(s, dev):
        import importlib
        repo = patched_iv2s_repo(); sys.path.insert(0, os.path.dirname(repo))  # import as a plain package (no HF dynamic modules)
        pkg = os.path.basename(repo); open(f"{repo}/__init__.py", "a").close()
        mod = importlib.import_module(f"{pkg}.modeling_internvideo2encoder")
        cfg = mod.config.from_pretrained(repo)
        for k in ("use_flash_attn", "use_fused_rmsnorm", "use_fused_mlp"): cfg.model.vision_encoder[k] = False
        cfg.device = dev
        s.m = mod.InternVideo2_CLIP_small.from_pretrained(repo, config=cfg, torch_dtype=torch.float32).to(dev).eval()
        if dev == "cuda": s.m = s.m.half()
        s.dev = dev; s.mean, s.std = MEAN_IN, STD_IN
    def video(s, x): return s.m.encode_vision(x.to(s.dev, dtype=next(s.m.parameters()).dtype), test=True).float()  # x [B,T,C,H,W]
    def text(s, caps): return s.m.encode_text(s.m.tokenizer(caps).to(s.dev)).float()

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

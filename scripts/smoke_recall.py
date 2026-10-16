#!/usr/bin/env python3
"""
P3 smoke: exact text-to-video R@1 / R@5 / MedR
Conditions: clean, pad_dup, pad_junk (budget=8, k=4)
Frozen model. Prefer InternVideo2 stage-2 small; fallback to ViCLIP-B-16-hf.
Do not train.
"""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2

from frame_sampler import sample_frames

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest.json"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BUDGET = 8


def count_frames(path):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(n, 1)


def extract_frames(path, indices):
    """Extract frames at given indices (0-based). Returns list of RGB PIL Images."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    cap.release()
    return frames


def load_model():
    """Try InternVideo2 stage-2 small; fall back to ViCLIP-B-16-hf."""
    # Ensure pkg_resources is available (ViCLIP modeling file needs it)
    try:
        import pkg_resources
    except ImportError:
        import subprocess, sys
        print("Installing setuptools for pkg_resources...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "setuptools"])
        import pkg_resources
    model_name = None
    try:
        from transformers import AutoModel, AutoTokenizer
        # Smallest practical stage-2 that may fit 16GB
        # InternVideo2-Stage2_1B is large; try distillation or ViCLIP first
        print("Attempting ViCLIP-B-16-hf (fits 16GB T4)...")
        model = AutoModel.from_pretrained(
            "OpenGVLab/ViCLIP-B-16-hf", trust_remote_code=True
        ).to(DEVICE).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            "OpenGVLab/ViCLIP-B-16-hf", trust_remote_code=True
        )
        model_name = "ViCLIP-B-16-hf"
        print(f"Loaded {model_name}")
        return model, tokenizer, model_name
    except Exception as e:
        print(f"ViCLIP load failed: {e}")
        print("BLOCKER: no suitable frozen model loaded. Install transformers + trust_remote_code deps.")
        return None, None, None


def get_video_embedding(model, frames, model_name):
    """Return L2-normalized video embedding."""
    if model_name.startswith("ViCLIP"):
        imgs = []
        for fr in frames:
            im = fr.resize((224, 224))
            arr = np.array(im).astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            arr = (arr - mean) / std
            imgs.append(arr)
        tube = np.stack(imgs, axis=0)  # T,H,W,C
        tube = np.transpose(tube, (0, 3, 1, 2))  # T,C,H,W
        tube = torch.from_numpy(tube).float().unsqueeze(0).to(DEVICE)  # 1,T,C,H,W
        with torch.no_grad():
            if hasattr(model, "get_vid_features"):
                feat = model.get_vid_features(tube)
            else:
                feat = model.vision_model(tube).pooler_output if hasattr(model, "vision_model") else tube.mean(dim=[2, 3, 4])
            feat = F.normalize(feat, dim=-1)
        return feat.squeeze(0).cpu().numpy()
    else:
        raise NotImplementedError(model_name)


def get_text_embedding(model, tokenizer, text, model_name):
    if model_name.startswith("ViCLIP"):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=77).to(DEVICE)
        with torch.no_grad():
            if hasattr(model, "get_text_features"):
                feat = model.get_text_features(inputs["input_ids"])
            else:
                feat = model.text_model(**inputs).pooler_output
            feat = F.normalize(feat, dim=-1)
        return feat.squeeze(0).cpu().numpy()
    else:
        raise NotImplementedError(model_name)


def compute_metrics(sim_matrix):
    """sim_matrix: [n_text, n_video] higher better. Diagonal is correct match for t2v."""
    n = sim_matrix.shape[0]
    ranks = []
    for i in range(n):
        row = sim_matrix[i]
        rank = (row > row[i]).sum() + 1  # 1-based rank of the correct video
        ranks.append(rank)
    ranks = np.array(ranks)
    r1 = (ranks <= 1).mean()
    r5 = (ranks <= 5).mean()
    medr = np.median(ranks)
    return {"R@1": float(r1), "R@5": float(r5), "MedR": float(medr), "ranks": ranks.tolist()}


def main():
    if not MANIFEST.exists():
        print("BLOCKER: data/manifest.json missing")
        return
    with open(MANIFEST) as f:
        manifest = json.load(f)
    n = len(manifest)
    print(f"N_CLIPS: {n}")

    model, tokenizer, model_name = load_model()
    if model is None:
        print("BLOCKER: model load failed")
        return

    conditions = ["clean", "pad_dup", "pad_junk"]
    all_results = {"model": model_name, "n_clips": n, "conditions": {}}

    # Precompute text embeddings (same for all conditions)
    captions = [item["caption"] for item in manifest]
    text_embs = []
    for cap in captions:
        text_embs.append(get_text_embedding(model, tokenizer, cap, model_name))
    text_embs = np.stack(text_embs)  # N, D

    for cond in conditions:
        print(f"\n=== Condition: {cond} ===")
        video_embs = []
        for i, item in enumerate(manifest):
            clip_path = ROOT / item["clip_path"]
            junk_path = ROOT / item["junk_clip_path"]
            total = count_frames(clip_path)
            junk_total = count_frames(junk_path)
            samples = sample_frames(total, junk_total)
            indices = samples[cond]
            if cond == "pad_junk":
                frames = extract_frames(junk_path, indices[:4]) + extract_frames(clip_path, indices[4:])
            else:
                frames = extract_frames(clip_path, indices)
            frames = frames[:BUDGET]
            while len(frames) < BUDGET:
                frames.append(frames[-1] if frames else Image.new("RGB", (224, 224)))
            emb = get_video_embedding(model, frames, model_name)
            video_embs.append(emb)
            print(f"  clip {i+1}/{n} done")
        video_embs = np.stack(video_embs)  # N, D

        sim = text_embs @ video_embs.T  # N, N
        metrics = compute_metrics(sim)
        all_results["conditions"][cond] = metrics
        print(f"{cond}: R@1={metrics['R@1']:.3f} R@5={metrics['R@5']:.3f} MedR={metrics['MedR']:.1f}")

    out_path = RESULTS / "smoke_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {out_path}")
    print("DONE")


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# P3 Block 1 GPU job: encoder forward passes only. Needs data/ (frames + manifest.json) copied from the CPU box.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install -q -r requirements.txt
export P3_DATA=${P3_DATA:-$PWD/data}
python - <<PY  # repoint manifest root to local data dir
import json,os; p=os.environ["P3_DATA"]+"/manifest.json"; m=json.load(open(p)); m["root"]=os.environ["P3_DATA"]; json.dump(m,open(p,"w"))
PY
for M in viclip_l openclip_meanpool; do python scripts/encode.py --model $M --manifest $P3_DATA/manifest.json --emb emb --bs 32; done
# InternVideo2_CLIP_S: public, flash_attn-free patched copy (SDPA), runs on T4/sm75 and CPU
if python scripts/encode.py --model internvideo2 --manifest $P3_DATA/manifest.json --emb emb --bs 16; then
  MODELS="internvideo2 viclip_l openclip_meanpool"
else echo "WARN: internvideo2 failed; scoring 2 models"; MODELS="viclip_l openclip_meanpool"; fi
python scripts/score.py --emb emb --models $MODELS --null openclip_meanpool --out results/block1.csv

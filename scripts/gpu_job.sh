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
if [[ -n "${IV2_DIR:-}" && -n "${IV2_CKPT:-}" && -n "${IV2_CFG:-}" ]]; then
  python scripts/encode.py --model internvideo2 --manifest $P3_DATA/manifest.json --emb emb --bs 32
  MODELS="internvideo2 viclip_l openclip_meanpool"
else echo "WARN: InternVideo2 env not set; skipping"; MODELS="viclip_l openclip_meanpool"; fi
python scripts/score.py --emb emb --models $MODELS --null openclip_meanpool --out results/block1.csv

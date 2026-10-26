#!/usr/bin/env bash
# P3 Block 2a GPU job: encode ONLY the 15,000 new rows (b2a_off1..4, b2a_split x seeds 0,1,2) for IV2-S and ViCLIP-L.
# off0 == Block 1 END is reused from Block 1 embeddings. openclip_meanpool is order-invariant and sees the same frame set in every
# Block 2a condition, so it equals Block 1 END exactly; set RUN_NULL=1 to encode it anyway as a sanity check (~+2 h on T4).
# Usage (repo root, after: tar xzf p3_block2a_bundle.tgz):  bash scripts/gpu_job_block2a.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install -q -r requirements.txt
export P3_DATA=${P3_DATA:-$PWD/data}; MAN=$P3_DATA/manifest_block2a.json
python - <<PY
import json,os; p="$MAN"; m=json.load(open(p)); m["root"]=os.environ["P3_DATA"]; json.dump(m,open(p,"w")); print("rows",len(m["rows"]))
PY
mkdir -p emb_b2a; S=$(date +%s)
python scripts/encode.py --model internvideo2 --out_name internvideo2_s --manifest $MAN --emb emb_b2a --bs 16 2>&1 | tee emb_b2a/iv2s.log
echo "iv2s_wall_s=$(( $(date +%s)-S ))" | tee emb_b2a/meta.txt; S=$(date +%s)
python scripts/encode.py --model viclip_l --manifest $MAN --emb emb_b2a --bs 32 2>&1 | tee emb_b2a/viclip_l.log
echo "viclip_l_wall_s=$(( $(date +%s)-S ))" | tee -a emb_b2a/meta.txt
if [[ "${RUN_NULL:-0}" == 1 ]]; then S=$(date +%s)
  python scripts/encode.py --model openclip_meanpool --manifest $MAN --emb emb_b2a --bs 32 2>&1 | tee emb_b2a/openclip.log
  echo "openclip_wall_s=$(( $(date +%s)-S ))" | tee -a emb_b2a/meta.txt; fi
git rev-parse HEAD > emb_b2a/repo_commit.txt; nvidia-smi > emb_b2a/nvidia_smi.txt || true
echo "Return emb_b2a/ (video_b2a_*.npy per model + logs). Scoring is done on the CPU box with scripts/score_block2a.py."

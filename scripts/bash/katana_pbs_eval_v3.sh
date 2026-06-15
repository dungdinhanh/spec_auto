#!/bin/bash
#PBS -q R7936711
#PBS -l select=1:ncpus=12:mem=128gb:ngpus=1:gpu_model=H200
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -N dflash_eval

# Per-run evaluation for the v3 sweep. Takes RUN_NAME via env.
# Runs three evals:
#   1. Cached-logits accuracy on train/val/test UUID splits (target_coc_outputs)
#   2. E2E on-shelf (200 test UUIDs)
#   3. E2E off-shelf (300 UUIDs sampled from PhysicalAI-AV val, seed=42)
#
# Writes three JSONs + one combined eval_results.json under
# $FLORA/runs/$RUN_NAME/eval/

set -e

export SCRATCH=/srv/scratch/z3552416
export FLORA=/srv/scratch/flora/dungda
export HF_HOME=$SCRATCH/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3
export VLM_PATH=$FLORA/models/Qwen3-VL-8B-Instruct
export PROCESSOR_PATH=$FLORA/models/Qwen3-VL-2B-Instruct

source $SCRATCH/envs/alpamayo/bin/activate

cd $PBS_O_WORKDIR
export PYTHONPATH=$PBS_O_WORKDIR/src:$PYTHONPATH

if [ -z "$RUN_NAME" ]; then
  echo "ERROR: RUN_NAME env var not set"
  exit 2
fi

RUN_DIR=$FLORA/runs/$RUN_NAME
EVAL_DIR=$RUN_DIR/eval
DRAFT=$RUN_DIR/draft_final.pt

mkdir -p $EVAL_DIR

echo "=== eval $RUN_NAME start $(date) ==="
echo "run_dir=$RUN_DIR"
echo "draft=$DRAFT"
if [ ! -f "$DRAFT" ]; then
  echo "ERROR: draft checkpoint missing: $DRAFT"
  ls -la $RUN_DIR 2>&1 || true
  exit 3
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --- 1. Cached-logits eval on target_coc_outputs (train/val/test via v3 UUIDs) ---
echo ""; echo "=== [1/3] cached-logits eval (train/val/test) ==="
python claude_mod/eval_draft_train_val_test.py \
    --target_path $FLORA/models/Alpamayo-R1-10B \
    --draft_path $DRAFT \
    --target_outputs_dir $FLORA/runs/target_coc_outputs \
    --use_mrope_draft \
    --val_uuids_file $FLORA/runs/splits/val_uuids_v3.json \
    --test_uuids_file $FLORA/runs/splits/test_uuids_v3.json \
    --train_eval_clips 500 --val_eval_clips 300 --test_eval_clips 200 \
    --output_json $EVAL_DIR/cached_trainvaltest.json \
  2>&1 | tail -80

# --- 2. E2E on-shelf (200 UUIDs) ---
echo ""; echo "=== [2/3] e2e on-shelf (200 UUIDs) ==="
python claude_mod/e2e_spec_test.py \
    --target_path $FLORA/models/Alpamayo-R1-10B \
    --draft_path $DRAFT \
    --clips_dir $FLORA/data/alpamayo_clips_onshelf \
    --uuids_file $FLORA/runs/splits/test_uuids_v3.json \
    --use_mrope_draft \
    --output_json $EVAL_DIR/e2e_onshelf.json \
  2>&1 | tail -80

# --- 3. E2E off-shelf (300 UUIDs) ---
OFFSHELF_DIR=$FLORA/data/alpamayo_clips_offshelf
OFFSHELF_UUIDS=$OFFSHELF_DIR/test_offshelf_uuids.json
echo ""; echo "=== [3/3] e2e off-shelf ==="
if [ -f "$OFFSHELF_UUIDS" ] && [ $(ls $OFFSHELF_DIR/*.pt 2>/dev/null | wc -l) -gt 0 ]; then
  python claude_mod/e2e_spec_test.py \
      --target_path $FLORA/models/Alpamayo-R1-10B \
      --draft_path $DRAFT \
      --clips_dir $OFFSHELF_DIR \
      --uuids_file $OFFSHELF_UUIDS \
      --use_mrope_draft \
      --output_json $EVAL_DIR/e2e_offshelf.json \
    2>&1 | tail -80
else
  echo "off-shelf clips not ready — skipping. dir=$OFFSHELF_DIR"
fi

# --- combine ---
python - <<PY
import json, os, pathlib
d = pathlib.Path("$EVAL_DIR")
out = {"run_name": "$RUN_NAME"}
for name, fname in [("cached","cached_trainvaltest.json"),
                    ("e2e_onshelf","e2e_onshelf.json"),
                    ("e2e_offshelf","e2e_offshelf.json")]:
    p = d/fname
    out[name] = json.load(open(p)) if p.exists() else None
json.dump(out, open(d/"eval_results.json","w"), indent=2)
print("combined ->", d/"eval_results.json")
PY

echo "=== eval $RUN_NAME done $(date) ==="

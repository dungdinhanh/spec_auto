#!/bin/bash
# Unified e2e_spec_test eval: 5 models × (val + test) splits.
# Off-shelf already done. All numbers from the same script for consistency.
set -e

LOCAL=/home/ubuntu/local_data
SCRIPTS=/home/ubuntu/katana_transfer/code/claude_mod
SPLITS=/home/ubuntu/katana_transfer/splits
OUTDIR=$LOCAL/runs/unified_e2e_eval
mkdir -p "$OUTDIR"

TARGET=$LOCAL/models/Alpamayo-R1-10B
COC=$LOCAL/runs/target_coc_outputs   # e2e_spec_test uses cached clips here
VAL=$SPLITS/val_uuids_v3.json
TEST=$SPLITS/test_uuids_v3.json

PYBIN=/home/ubuntu/alpamayo_env/bin/python
export PYTHONPATH=/home/ubuntu/dflash_code:/home/ubuntu/katana_transfer/code/src
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

# Drafts (5):
W3=$LOCAL/runs/dflash_L4_lr1e-4_ep15_bs16_warm_sharon1/draft_final.pt
W5P=$LOCAL/runs/dflash_L4_lr1e-4_ep15_bs16_warm_v5_partial_sharon1/draft_final.pt
V3AARL=$LOCAL/runs/rl_v3_optionA_K5_refAnchor/draft_final.pt
V4LR1=$LOCAL/runs/aarl_v4_filter_lr1e-6/draft_step_3000.pt
V4LR2=$LOCAL/runs/aarl_v4_filter_lr2e-6/draft_step_3000.pt

# Launch 10 jobs: (5 models × 2 splits), distribute round-robin across GPUs 0-7
JOBS=(
  "$W3|$VAL|warm_sharon1_val"
  "$W3|$TEST|warm_sharon1_test"
  "$W5P|$VAL|warm_v5partial_val"
  "$W5P|$TEST|warm_v5partial_test"
  "$V3AARL|$VAL|v3aarl_refAnchor_val"
  "$V3AARL|$TEST|v3aarl_refAnchor_test"
  "$V4LR1|$VAL|v4_filter_lr1e6_step3000_val"
  "$V4LR1|$TEST|v4_filter_lr1e6_step3000_test"
  "$V4LR2|$VAL|v4_filter_lr2e6_step3000_val"
  "$V4LR2|$TEST|v4_filter_lr2e6_step3000_test"
)

i=0
for j in "${JOBS[@]}"; do
  IFS='|' read -r draft uuids tag <<< "$j"
  gpu=$((i % 8))
  i=$((i + 1))
  log=$OUTDIR/${tag}.log
  json=$OUTDIR/${tag}.json
  CUDA_VISIBLE_DEVICES=$gpu nohup $PYBIN $SCRIPTS/e2e_spec_test.py \
    --target_path $TARGET --draft_path $draft \
    --clips_dir $COC --uuids_file $uuids \
    --num_draft_layers 4 --block_size 16 \
    --output_json $json > $log 2>&1 &
  echo "GPU $gpu launched (pid=$!) tag=$tag"
done

sleep 2
ps -ef | grep e2e_spec_test | grep -v grep | wc -l
wait
echo "UNIFIED_E2E_DONE $(date)"

#!/bin/bash
# Off-shelf (PhysicalAI-AV val, ~300 clips) eval for v4 AARL filter ckpts.
# Uses claude_mod/e2e_spec_test.py. 6 ckpts × 1 split distributed across GPUs 0-5.
set -e

LOCAL=/home/ubuntu/local_data
SCRIPTS=/home/ubuntu/katana_transfer/code/claude_mod
CLIPS=/home/ubuntu/katana_transfer/offshelf/alpamayo_clips_offshelf
UUIDS=$CLIPS/test_offshelf_uuids.json
OUTDIR=$LOCAL/runs/aarl_v4_filter_offshelf_eval
mkdir -p "$OUTDIR"

TARGET=$LOCAL/models/Alpamayo-R1-10B
PYBIN=/home/ubuntu/alpamayo_env/bin/python
export PYTHONPATH=/home/ubuntu/dflash_code:/home/ubuntu/katana_transfer/code/src
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

# 6 ckpts: (lr1e6 × {500, 2500, 3000}) + (lr2e6 × {500, 2500, 3000})
declare -a JOBS=(
  "0|aarl_v4_filter_lr1e-6|draft_step_500.pt|lr1e6_step500"
  "1|aarl_v4_filter_lr1e-6|draft_step_2500.pt|lr1e6_step2500"
  "2|aarl_v4_filter_lr1e-6|draft_step_3000.pt|lr1e6_step3000"
  "3|aarl_v4_filter_lr2e-6|draft_step_500.pt|lr2e6_step500"
  "4|aarl_v4_filter_lr2e-6|draft_step_2500.pt|lr2e6_step2500"
  "5|aarl_v4_filter_lr2e-6|draft_step_3000.pt|lr2e6_step3000"
)

for j in "${JOBS[@]}"; do
  IFS='|' read -r gpu rundir ckpt tag <<< "$j"
  draft=$LOCAL/runs/$rundir/$ckpt
  log=$OUTDIR/${tag}.log
  json=$OUTDIR/${tag}.json
  CUDA_VISIBLE_DEVICES=$gpu nohup $PYBIN $SCRIPTS/e2e_spec_test.py \
    --target_path $TARGET --draft_path $draft \
    --clips_dir $CLIPS --uuids_file $UUIDS \
    --num_draft_layers 4 --block_size 16 \
    --output_json $json > $log 2>&1 &
  echo "GPU $gpu launched (pid=$!) tag=$tag log=$log"
done
sleep 1
ps -ef | grep e2e_spec_test | grep -v grep | awk '{print $2}'
wait
echo "OFFSHELF_EVAL_V4_DONE $(date)"

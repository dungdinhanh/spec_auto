#!/bin/bash
# 4-GPU parallel eval of focused-CE ckpts (top-3 by val_quick + draft_final + per-epoch).
# val_v3 (300) + test_v3 (200). Output to dflash_L4_focused_ce_warm_init_eval/.
# Runs on GPUs 0-3 (K=30 RL is on GPUs 4-7).
set -e

LOCAL=/home/ubuntu/local_data
SCRIPTS=/home/ubuntu/katana_transfer/code/scripts
SPLITS=/home/ubuntu/katana_transfer/splits
OUTDIR=$LOCAL/runs/dflash_L4_focused_ce_warm_init_eval
mkdir -p "$OUTDIR"

TARGET=$LOCAL/models/Alpamayo-R1-10B
COC=$LOCAL/runs/target_coc_outputs
VAL=$SPLITS/val_uuids_v3.json
TEST=$SPLITS/test_uuids_v3.json

D=$LOCAL/runs/dflash_L4_focused_ce_warm_init

CK_GPU0=( $D/draft_final.pt $D/draft_topk_step_400_wce1.3796.pt $D/draft_topk_step_475_wce1.3925.pt )
CK_GPU1=( $D/draft_topk_step_800_wce1.5090.pt $D/draft_epoch_1.pt )
CK_GPU2=( $D/draft_epoch_2.pt $D/draft_step_500.pt )
CK_GPU3=( $D/draft_epoch_3.pt $D/draft_step_1000.pt )

PYBIN=/home/ubuntu/alpamayo_env/bin/python
export PYTHONPATH=/home/ubuntu/dflash_code:/home/ubuntu/katana_transfer/code/src

run_chunk() {
  local gpu=$1; shift
  local drafts=("$@")
  local log=$OUTDIR/eval_gpu${gpu}.log
  local csv=$OUTDIR/eval_gpu${gpu}.csv
  CUDA_VISIBLE_DEVICES=$gpu nohup $PYBIN $SCRIPTS/eval_ckpt_sweep_vt.py \
    --target_path $TARGET \
    --drafts "${drafts[@]}" \
    --target_outputs_dir $COC \
    --val_uuids_file $VAL \
    --test_uuids_file $TEST \
    --output_csv $csv \
    > $log 2>&1 &
  echo "GPU $gpu launched (pid=$!)  log=$log"
}

run_chunk 0 "${CK_GPU0[@]}"
run_chunk 1 "${CK_GPU1[@]}"
run_chunk 2 "${CK_GPU2[@]}"
run_chunk 3 "${CK_GPU3[@]}"
sleep 1
ps -ef | grep eval_ckpt_sweep_vt | grep -v grep | awk '{print $2}' | head -10

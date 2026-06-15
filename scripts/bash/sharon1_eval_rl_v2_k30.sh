#!/bin/bash
# 4-GPU parallel eval of K=30 saved ckpts (8 topk + draft_final = 9 total).
# val_v3 (300) + test_v3 (200). Output to rl_v2_N5_k30_eval/.
# Runs on GPUs 4-7 (v3 Option A is on GPUs 0-3).
set -e

LOCAL=/home/ubuntu/local_data
SCRIPTS=/home/ubuntu/katana_transfer/code/scripts
SPLITS=/home/ubuntu/katana_transfer/splits
OUTDIR=$LOCAL/runs/rl_v2_N5_k30_eval
mkdir -p "$OUTDIR"

TARGET=$LOCAL/models/Alpamayo-R1-10B
COC=$LOCAL/runs/target_coc_outputs
VAL=$SPLITS/val_uuids_v3.json
TEST=$SPLITS/test_uuids_v3.json

K30=$LOCAL/runs/rl_v2_N5_k30

# Available ckpts: draft_final, 500, 750, 1250, 1500, 2000, 2250, 2500, 3500.
# Top-1 by rolling acc_rate is step 1500 (0.864). Distribute 9 ckpts × 4 GPUs.
CK_GPU4=( $K30/draft_final.pt $K30/draft_step_500.pt $K30/draft_step_750.pt )
CK_GPU5=( $K30/draft_step_1250.pt $K30/draft_step_1500.pt )
CK_GPU6=( $K30/draft_step_2000.pt $K30/draft_step_2250.pt )
CK_GPU7=( $K30/draft_step_2500.pt $K30/draft_step_3500.pt )

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

run_chunk 4 "${CK_GPU4[@]}"
run_chunk 5 "${CK_GPU5[@]}"
run_chunk 6 "${CK_GPU6[@]}"
run_chunk 7 "${CK_GPU7[@]}"
sleep 1
ps -ef | grep eval_ckpt_sweep_vt | grep -v grep | awk '{print $2}' | head -10

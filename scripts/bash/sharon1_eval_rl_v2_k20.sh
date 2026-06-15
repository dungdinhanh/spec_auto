#!/bin/bash
# 4-GPU parallel eval of K=20 saved ckpts (8 topk + draft_final = 9 total).
# val_v3 (300) + test_v3 (200). Output to rl_v2_N5_k20_eval/.
set -e

LOCAL=/home/ubuntu/local_data
SCRIPTS=/home/ubuntu/katana_transfer/code/scripts
SPLITS=/home/ubuntu/katana_transfer/splits
OUTDIR=$LOCAL/runs/rl_v2_N5_k20_eval
mkdir -p "$OUTDIR"

TARGET=$LOCAL/models/Alpamayo-R1-10B
COC=$LOCAL/runs/target_coc_outputs
VAL=$SPLITS/val_uuids_v3.json
TEST=$SPLITS/test_uuids_v3.json

K20=$LOCAL/runs/rl_v2_N5_k20

CK_GPU0=( $K20/draft_final.pt $K20/draft_step_500.pt $K20/draft_step_750.pt )
CK_GPU1=( $K20/draft_step_1250.pt $K20/draft_step_1500.pt )
CK_GPU2=( $K20/draft_step_2000.pt $K20/draft_step_2250.pt )
CK_GPU3=( $K20/draft_step_2500.pt $K20/draft_step_3500.pt )

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

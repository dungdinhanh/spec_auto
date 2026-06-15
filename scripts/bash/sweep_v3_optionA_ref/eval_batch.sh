#!/bin/bash
# Eval one ref-anchor sweep run across 4 GPUs.
# Usage: eval_batch.sh <run_dirname> <gpu0> <gpu1> <gpu2> <gpu3>
set -e
RUN_NAME=$1
G0=$2 G1=$3 G2=$4 G3=$5

LOCAL=/home/ubuntu/local_data
SCRIPTS=/home/ubuntu/katana_transfer/code/scripts
SPLITS=/home/ubuntu/katana_transfer/splits
OUTDIR=$LOCAL/runs/${RUN_NAME}_eval
mkdir -p "$OUTDIR"

TARGET=$LOCAL/models/Alpamayo-R1-10B
COC=$LOCAL/runs/target_coc_outputs
VAL=$SPLITS/val_uuids_v3.json
TEST=$SPLITS/test_uuids_v3.json
RUN=$LOCAL/runs/$RUN_NAME

# 12 ckpts spread across 4 GPUs = 3 ckpts per GPU.
CK_G0=( $RUN/draft_final.pt $RUN/draft_step_500.pt $RUN/draft_step_1000.pt )
CK_G1=( $RUN/draft_step_1500.pt $RUN/draft_step_2000.pt $RUN/draft_step_2500.pt )
CK_G2=( $RUN/draft_step_3000.pt $RUN/draft_step_3500.pt $RUN/draft_step_4000.pt )
CK_G3=( $RUN/draft_step_4500.pt $RUN/draft_step_5000.pt $RUN/draft_step_5500.pt )

PYBIN=/home/ubuntu/alpamayo_env/bin/python
export PYTHONPATH=/home/ubuntu/dflash_code:/home/ubuntu/katana_transfer/code/src

run_chunk() {
  local gpu=$1; shift
  local drafts=("$@")
  local log=$OUTDIR/eval_gpu${gpu}.log
  local csv=$OUTDIR/eval_gpu${gpu}.csv
  CUDA_VISIBLE_DEVICES=$gpu nohup $PYBIN $SCRIPTS/eval_ckpt_sweep_vt.py \
    --target_path $TARGET --drafts "${drafts[@]}" \
    --target_outputs_dir $COC \
    --val_uuids_file $VAL --test_uuids_file $TEST \
    --output_csv $csv > $log 2>&1 &
  echo "$RUN_NAME GPU $gpu launched (pid=$!)  log=$log"
}

run_chunk $G0 "${CK_G0[@]}"
run_chunk $G1 "${CK_G1[@]}"
run_chunk $G2 "${CK_G2[@]}"
run_chunk $G3 "${CK_G3[@]}"

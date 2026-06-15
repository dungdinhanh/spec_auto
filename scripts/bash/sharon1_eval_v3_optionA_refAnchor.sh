#!/bin/bash
# Eval v3 Option A (ref-anchor) on GPUs 4-7.
# 12 ckpts × 2 splits = 24 rows. Output: rl_v3_optionA_K5_refAnchor_eval/.
set -e

LOCAL=/home/ubuntu/local_data
SCRIPTS=/home/ubuntu/katana_transfer/code/scripts
SPLITS=/home/ubuntu/katana_transfer/splits
OUTDIR=$LOCAL/runs/rl_v3_optionA_K5_refAnchor_eval
mkdir -p "$OUTDIR"

TARGET=$LOCAL/models/Alpamayo-R1-10B
COC=$LOCAL/runs/target_coc_outputs
VAL=$SPLITS/val_uuids_v3.json
TEST=$SPLITS/test_uuids_v3.json
RUN=$LOCAL/runs/rl_v3_optionA_K5_refAnchor

# 12 ckpts spread across 4 GPUs = 3 ckpts per GPU
CK_GPU4=( $RUN/draft_final.pt $RUN/draft_step_500.pt $RUN/draft_step_1000.pt )
CK_GPU5=( $RUN/draft_step_1500.pt $RUN/draft_step_2000.pt $RUN/draft_step_2500.pt )
CK_GPU6=( $RUN/draft_step_3000.pt $RUN/draft_step_3500.pt $RUN/draft_step_4000.pt )
CK_GPU7=( $RUN/draft_step_4500.pt $RUN/draft_step_5000.pt $RUN/draft_step_5500.pt )

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
  echo "GPU $gpu launched (pid=$!)  log=$log"
}

run_chunk 4 "${CK_GPU4[@]}"
run_chunk 5 "${CK_GPU5[@]}"
run_chunk 6 "${CK_GPU6[@]}"
run_chunk 7 "${CK_GPU7[@]}"
sleep 1
ps -ef | grep eval_ckpt_sweep_vt | grep -v grep | awk '{print $2}' | head

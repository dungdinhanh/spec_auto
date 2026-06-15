#!/bin/bash
# Eval one K-sweep training run on 4 GPUs.
# Usage: eval.sh <K_label> <gpu0> <gpu1> <gpu2> <gpu3>
# Where K_label is e.g. K10/K15/K20, matching dir name suffix.
set -e
K=$1
G0=$2; G1=$3; G2=$4; G3=$5

LOCAL=/home/ubuntu/local_data
SCRIPTS=/home/ubuntu/katana_transfer/code/scripts
SPLITS=/home/ubuntu/katana_transfer/splits
RUN_NAME=rl_v3_optA_ref_$K
RUN=$LOCAL/runs/$RUN_NAME
OUTDIR=$LOCAL/runs/${RUN_NAME}_eval
mkdir -p "$OUTDIR"

TARGET=$LOCAL/models/Alpamayo-R1-10B
COC=$LOCAL/runs/target_coc_outputs
VAL=$SPLITS/val_uuids_v3.json
TEST=$SPLITS/test_uuids_v3.json

# 12 ckpts (final + 500..5500) spread across 4 GPUs = 3 ckpts/GPU.
# If draft_final is missing (training crashed at end), it'll just fail silently.
CK_G0=( $RUN/draft_final.pt $RUN/draft_step_500.pt $RUN/draft_step_1000.pt )
CK_G1=( $RUN/draft_step_1500.pt $RUN/draft_step_2000.pt $RUN/draft_step_2500.pt )
CK_G2=( $RUN/draft_step_3000.pt $RUN/draft_step_3500.pt $RUN/draft_step_4000.pt )
CK_G3=( $RUN/draft_step_4500.pt $RUN/draft_step_5000.pt $RUN/draft_step_5500.pt )

PYBIN=/home/ubuntu/alpamayo_env/bin/python
export PYTHONPATH=/home/ubuntu/dflash_code:/home/ubuntu/katana_transfer/code/src

run_chunk() {
  local gpu=$1; shift
  local drafts=("$@")
  # Filter out missing files
  local existing=()
  for d in "${drafts[@]}"; do
    [ -f "$d" ] && existing+=("$d")
  done
  if [ ${#existing[@]} -eq 0 ]; then return; fi
  local log=$OUTDIR/eval_gpu${gpu}.log
  local csv=$OUTDIR/eval_gpu${gpu}.csv
  CUDA_VISIBLE_DEVICES=$gpu nohup $PYBIN $SCRIPTS/eval_ckpt_sweep_vt.py \
    --target_path $TARGET --drafts "${existing[@]}" \
    --target_outputs_dir $COC \
    --val_uuids_file $VAL --test_uuids_file $TEST \
    --output_csv $csv > $log 2>&1 &
  echo "$RUN_NAME GPU $gpu launched (pid=$!) [${#existing[@]} ckpts]"
}

run_chunk $G0 "${CK_G0[@]}"
run_chunk $G1 "${CK_G1[@]}"
run_chunk $G2 "${CK_G2[@]}"
run_chunk $G3 "${CK_G3[@]}"

# Wait for all 4 evals to finish before returning (so coordinator can chain)
wait
echo "$RUN_NAME eval done $(date)"

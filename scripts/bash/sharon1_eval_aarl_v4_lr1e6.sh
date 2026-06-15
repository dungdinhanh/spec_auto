#!/bin/bash
# Eval v4 AARL filter run lr=1e-6 on GPUs 0-3.
# Run AFTER training PID dies (chained from launch).
set -e

LOCAL=/home/ubuntu/local_data
SCRIPTS=/home/ubuntu/katana_transfer/code/scripts
SPLITS=/home/ubuntu/katana_transfer/splits
OUTDIR=$LOCAL/runs/aarl_v4_filter_lr1e-6_eval
mkdir -p "$OUTDIR"

TARGET=$LOCAL/models/Alpamayo-R1-10B
COC=$LOCAL/runs/target_coc_outputs
VAL=$SPLITS/val_uuids_v3.json
TEST=$SPLITS/test_uuids_v3.json
RUN=$LOCAL/runs/aarl_v4_filter_lr1e-6

# Discover available ckpts (skip if missing). Spread across 4 GPUs.
CKPTS=( $RUN/draft_final.pt $RUN/draft_step_500.pt $RUN/draft_step_1000.pt \
        $RUN/draft_step_1500.pt $RUN/draft_step_2000.pt $RUN/draft_step_2500.pt \
        $RUN/draft_step_3000.pt $RUN/draft_step_3500.pt $RUN/draft_step_4000.pt \
        $RUN/draft_step_4500.pt $RUN/draft_step_5000.pt $RUN/draft_step_5500.pt )
EXIST=()
for c in "${CKPTS[@]}"; do [ -f "$c" ] && EXIST+=("$c"); done
N=${#EXIST[@]}
echo "Found $N ckpts"

PYBIN=/home/ubuntu/alpamayo_env/bin/python
export PYTHONPATH=/home/ubuntu/dflash_code:/home/ubuntu/katana_transfer/code/src

# Round-robin distribute ckpts across GPUs 0-3
declare -a G0 G1 G2 G3
for i in "${!EXIST[@]}"; do
  case $((i % 4)) in
    0) G0+=("${EXIST[$i]}") ;;
    1) G1+=("${EXIST[$i]}") ;;
    2) G2+=("${EXIST[$i]}") ;;
    3) G3+=("${EXIST[$i]}") ;;
  esac
done

run_chunk() {
  local gpu=$1; shift
  local drafts=("$@")
  [ ${#drafts[@]} -eq 0 ] && return
  local log=$OUTDIR/eval_gpu${gpu}.log
  local csv=$OUTDIR/eval_gpu${gpu}.csv
  CUDA_VISIBLE_DEVICES=$gpu nohup $PYBIN $SCRIPTS/eval_ckpt_sweep_vt.py \
    --target_path $TARGET --drafts "${drafts[@]}" \
    --target_outputs_dir $COC \
    --val_uuids_file $VAL --test_uuids_file $TEST \
    --output_csv $csv > $log 2>&1 &
  echo "GPU $gpu launched (pid=$!) ckpts=${#drafts[@]} log=$log"
}

run_chunk 0 "${G0[@]}"
run_chunk 1 "${G1[@]}"
run_chunk 2 "${G2[@]}"
run_chunk 3 "${G3[@]}"
sleep 1
ps -ef | grep eval_ckpt_sweep_vt | grep -v grep | awk '{print $2}' | head
wait
echo "EVAL_V4_LR1E6_DONE $(date)"

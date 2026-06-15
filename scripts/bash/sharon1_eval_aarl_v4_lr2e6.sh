#!/bin/bash
# Eval v4 AARL filter run lr=2e-6 on GPUs 4-7.
set -e

LOCAL=/home/ubuntu/local_data
SCRIPTS=/home/ubuntu/katana_transfer/code/scripts
SPLITS=/home/ubuntu/katana_transfer/splits
OUTDIR=$LOCAL/runs/aarl_v4_filter_lr2e-6_eval
mkdir -p "$OUTDIR"

TARGET=$LOCAL/models/Alpamayo-R1-10B
COC=$LOCAL/runs/target_coc_outputs
VAL=$SPLITS/val_uuids_v3.json
TEST=$SPLITS/test_uuids_v3.json
RUN=$LOCAL/runs/aarl_v4_filter_lr2e-6

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

declare -a G4 G5 G6 G7
for i in "${!EXIST[@]}"; do
  case $((i % 4)) in
    0) G4+=("${EXIST[$i]}") ;;
    1) G5+=("${EXIST[$i]}") ;;
    2) G6+=("${EXIST[$i]}") ;;
    3) G7+=("${EXIST[$i]}") ;;
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

run_chunk 4 "${G4[@]}"
run_chunk 5 "${G5[@]}"
run_chunk 6 "${G6[@]}"
run_chunk 7 "${G7[@]}"
sleep 1
ps -ef | grep eval_ckpt_sweep_vt | grep -v grep | awk '{print $2}' | head
wait
echo "EVAL_V4_LR2E6_DONE $(date)"

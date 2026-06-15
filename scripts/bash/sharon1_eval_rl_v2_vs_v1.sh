#!/bin/bash
# 4-GPU parallel eval sweep on sharon1 GPUs 4-7:
# v1 baseline (rl_e2v2_N5 step_750) vs v2 RL top-8 saved ckpts.
# val (300) + test (200) on v3 splits.
set -e

LOCAL=/home/ubuntu/local_data
SCRIPTS=/home/ubuntu/katana_transfer/code/scripts
SPLITS=/home/ubuntu/katana_transfer/splits
OUTDIR=$LOCAL/runs/rl_e2v2_v2_N5_eval
mkdir -p "$OUTDIR"

TARGET=$LOCAL/models/Alpamayo-R1-10B
COC=$LOCAL/runs/target_coc_outputs
VAL=$SPLITS/val_uuids_v3.json
TEST=$SPLITS/test_uuids_v3.json

# v1 baseline
V1=$LOCAL/runs/rl_e2v2_N5/draft_step_750.pt
# v2 RL ckpts
V2DIR=$LOCAL/runs/rl_e2v2_v2_N5

CK_GPU4=( $V1 $V2DIR/draft_step_500.pt $V2DIR/draft_step_750.pt )
CK_GPU5=( $V2DIR/draft_step_1250.pt $V2DIR/draft_step_1500.pt )
CK_GPU6=( $V2DIR/draft_step_2000.pt $V2DIR/draft_step_2250.pt )
CK_GPU7=( $V2DIR/draft_step_2500.pt $V2DIR/draft_step_3500.pt )

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
echo "----"
ps -ef | grep eval_ckpt_sweep_vt | grep -v grep | awk '{print $2,$NF}' | head -10

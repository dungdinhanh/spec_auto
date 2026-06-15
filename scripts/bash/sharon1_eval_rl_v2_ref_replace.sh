#!/bin/bash
# 4-GPU parallel eval of moving-reference (periodic ref-replace, every 1000 steps) ckpts.
# 9 ckpts (final + 8 step ckpts) on val_v3 (300) + test_v3 (200). GPUs 4-7.
set -e

LOCAL=/home/ubuntu/local_data
SCRIPTS=/home/ubuntu/katana_transfer/code/scripts
SPLITS=/home/ubuntu/katana_transfer/splits
OUTDIR=$LOCAL/runs/rl_v2_N5_K4_ref_replace_1000_eval
mkdir -p "$OUTDIR"

TARGET=$LOCAL/models/Alpamayo-R1-10B
COC=$LOCAL/runs/target_coc_outputs
VAL=$SPLITS/val_uuids_v3.json
TEST=$SPLITS/test_uuids_v3.json

REF=$LOCAL/runs/rl_v2_N5_K4_ref_replace_1000

CK_GPU4=( $REF/draft_final.pt $REF/draft_step_500.pt $REF/draft_step_750.pt )
CK_GPU5=( $REF/draft_step_1250.pt $REF/draft_step_1500.pt )
CK_GPU6=( $REF/draft_step_2000.pt $REF/draft_step_2250.pt )
CK_GPU7=( $REF/draft_step_2500.pt $REF/draft_step_3500.pt )

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

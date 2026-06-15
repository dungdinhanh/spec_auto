#!/bin/bash
# Evaluate 4 key ckpts on a 100-clip train sample (held inside the training pool)
# to validate the "RL overfits training distribution" hypothesis.
# Reuses eval_ckpt_sweep_vt.py with train100 as val_uuids_file.
# The "val" column in the output CSV = train100 results.
set -e

LOCAL=/home/ubuntu/local_data
SCRIPTS=/home/ubuntu/katana_transfer/code/scripts
SPLITS=/home/ubuntu/katana_transfer/splits
OUTDIR=$LOCAL/runs/rl_e2v2_v2_N5_eval
mkdir -p "$OUTDIR"

TARGET=$LOCAL/models/Alpamayo-R1-10B
COC=$LOCAL/runs/target_coc_outputs
TRAIN100=$SPLITS/train100_uuids_v3.json
TEST=$SPLITS/test_uuids_v3.json   # for cross-check vs prior eval (test L should match)

CK_GPU4=$LOCAL/runs/dflash_L4_lr1e-4_ep15_bs16_warm_sharon1/draft_final.pt
CK_GPU5=$LOCAL/runs/rl_e2v2_N5/draft_step_750.pt
CK_GPU6=$LOCAL/runs/rl_e2v2_v2_N5/draft_step_1500.pt
CK_GPU7=$LOCAL/runs/rl_e2v2_v2_N5/draft_step_2500.pt

PYBIN=/home/ubuntu/alpamayo_env/bin/python
export PYTHONPATH=/home/ubuntu/dflash_code:/home/ubuntu/katana_transfer/code/src

run_chunk() {
  local gpu=$1; local draft=$2; local label=$3
  local log=$OUTDIR/eval_train100_gpu${gpu}_${label}.log
  local csv=$OUTDIR/eval_train100_gpu${gpu}_${label}.csv
  CUDA_VISIBLE_DEVICES=$gpu nohup $PYBIN $SCRIPTS/eval_ckpt_sweep_vt.py \
    --target_path $TARGET \
    --drafts $draft \
    --target_outputs_dir $COC \
    --val_uuids_file $TRAIN100 \
    --test_uuids_file $TEST \
    --output_csv $csv \
    > $log 2>&1 &
  echo "GPU $gpu launched (pid=$!) ${label}  log=$log"
}

run_chunk 4 $CK_GPU4 noRLwarm
run_chunk 5 $CK_GPU5 v1RLpeak
run_chunk 6 $CK_GPU6 v2RL_step1500
run_chunk 7 $CK_GPU7 v2RL_step2500
sleep 1
ps -ef | grep eval_ckpt_sweep_vt | grep -v grep | awk '{print $2}' | head -10

#!/bin/bash
# Eval the two no-RL SFT ckpts (warm + random) on v3 val+test splits.
# Runs on freed GPUs 5 and 6 while GPU 4 is still finishing its v2 RL chunk.
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

WARM=$LOCAL/runs/dflash_L4_lr1e-4_ep15_bs16_warm_sharon1/draft_final.pt
RANDOM_INIT=$LOCAL/runs/dflash_L4_lr1e-4_ep15_bs16_sharon1/draft_final.pt

PYBIN=/home/ubuntu/alpamayo_env/bin/python
export PYTHONPATH=/home/ubuntu/dflash_code:/home/ubuntu/katana_transfer/code/src

run_chunk() {
  local gpu=$1; shift
  local drafts=("$@")
  local log=$OUTDIR/eval_norl_gpu${gpu}.log
  local csv=$OUTDIR/eval_norl_gpu${gpu}.csv
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

run_chunk 5 "$WARM"
run_chunk 6 "$RANDOM_INIT"
sleep 1
ps -ef | grep eval_ckpt_sweep_vt | grep -v grep | awk '{print $2,$NF}' | head -10

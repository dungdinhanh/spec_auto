#!/bin/bash
# Eval SFT v2 (warm + topk) ckpts on v3 val+test splits, in parallel on GPUs 4-7.
# Output: rl_e2v2_v2_N5_eval/eval_sft_v2_gpu{4-7}.csv
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
V2DIR=$LOCAL/runs/dflash_L4_lr1e-4_ep15_bs16_v2_sharon1

CK_GPU4=$V2DIR/draft_topk_step_4141_wce0.2709.pt   # current best by val_quick
CK_GPU5=$V2DIR/draft_topk_step_6600_wce0.2822.pt
CK_GPU6=$V2DIR/draft_topk_step_6882_wce0.2813.pt
CK_GPU7=$V2DIR/draft_final.pt

PYBIN=/home/ubuntu/alpamayo_env/bin/python
export PYTHONPATH=/home/ubuntu/dflash_code:/home/ubuntu/katana_transfer/code/src

run_chunk() {
  local gpu=$1; local draft=$2; local label=$3
  local log=$OUTDIR/eval_sft_v2_gpu${gpu}_${label}.log
  local csv=$OUTDIR/eval_sft_v2_gpu${gpu}_${label}.csv
  CUDA_VISIBLE_DEVICES=$gpu nohup $PYBIN $SCRIPTS/eval_ckpt_sweep_vt.py \
    --target_path $TARGET \
    --drafts $draft \
    --target_outputs_dir $COC \
    --val_uuids_file $VAL \
    --test_uuids_file $TEST \
    --output_csv $csv \
    > $log 2>&1 &
  echo "GPU $gpu ${label} (pid=$!)  log=$log"
}

run_chunk 4 $CK_GPU4 sft_v2_step4141
run_chunk 5 $CK_GPU5 sft_v2_step6600
run_chunk 6 $CK_GPU6 sft_v2_step6882
run_chunk 7 $CK_GPU7 sft_v2_final
sleep 1
ps -ef | grep eval_ckpt_sweep_vt | grep -v grep | awk '{print $2,$NF}' | head -10

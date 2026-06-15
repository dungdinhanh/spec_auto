#!/bin/bash
# Launch RL v2 N=5 with k_samples ∈ {20, 25, 30}, chunk_K=10, on sharon1.
# Round 1 (parallel): K=20 on GPUs 0-3 (port 29641) + K=25 on GPUs 4-7 (port 29642).
# K=30 will be launched separately on freed GPUs after one of round 1 completes.
set -e

LOCAL=/home/ubuntu/local_data
SCRIPTS=/home/ubuntu/katana_transfer/code/scripts
SPLITS=/home/ubuntu/katana_transfer/splits

INIT=$LOCAL/runs/dflash_L4_lr1e-4_ep15_bs16_warm_sharon1/draft_final.pt
TARGET=$LOCAL/models/Alpamayo-R1-10B
COC=$LOCAL/runs/target_coc_outputs
VAL=$SPLITS/val_uuids_v3.json
TEST=$SPLITS/test_uuids_v3.json

source /home/ubuntu/alpamayo_env/bin/activate
export PYTHONPATH=/home/ubuntu/katana_transfer/code/src
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

run_train() {
  local K=$1
  local GPUS=$2
  local PORT=$3
  local OUT=$LOCAL/runs/rl_v2_N5_k${K}
  local LOG=/tmp/rl_v2_N5_k${K}.log
  mkdir -p $OUT
  CUDA_VISIBLE_DEVICES=$GPUS nohup torchrun --nproc_per_node=4 --master_port=$PORT \
    $SCRIPTS/train_dflash_rl_action_v2.py \
    --target_path $TARGET \
    --init_draft_path $INIT \
    --target_outputs_dir $COC \
    --val_uuids_file $VAL \
    --test_uuids_file $TEST \
    --output_dir $OUT \
    --num_epochs 2 --lr 1e-5 --kl_weight 0.02 \
    --k_samples $K --k_chunk_size 10 --temperature 1.0 \
    --contamination_N 5 --accept_bonus 0.0 --block_start_decay 0.8 \
    --log_interval 25 --save_interval 250 --topk_save 8 --no_wandb \
    > $LOG 2>&1 &
  echo "K=$K  GPUs=$GPUS  port=$PORT  pid=$!  log=$LOG  outdir=$OUT"
}

run_train 20 0,1,2,3 29641
run_train 25 4,5,6,7 29642
sleep 1
echo "----"
ps -ef | grep "train_dflash_rl_action_v2.py" | grep -v grep | awk '{print $2,$NF}' | head -10

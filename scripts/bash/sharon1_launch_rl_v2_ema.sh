#!/bin/bash
# RL v2 N=5 K=4 with periodic ref replacement every 1000 steps.
# Tests if a moving (replaced) reference helps vs the static ref of the original v2.
# Same hyperparams as v2 N=5 baseline + --ref_update_interval=1000 --ref_update_mode=replace.
# Runs on sharon1 GPUs 0-3 in parallel with K=30 on GPUs 4-7.
set -e

LOCAL=/home/ubuntu/local_data
SCRIPTS=/home/ubuntu/katana_transfer/code/scripts
SPLITS=/home/ubuntu/katana_transfer/splits

OUT=$LOCAL/runs/rl_v2_N5_K4_ref_replace_1000
LOG=/tmp/rl_v2_N5_K4_ref_replace_1000.log
mkdir -p $OUT

source /home/ubuntu/alpamayo_env/bin/activate
export PYTHONPATH=/home/ubuntu/katana_transfer/code/src
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup torchrun --nproc_per_node=4 --master_port=29646 \
  $SCRIPTS/train_dflash_rl_action_v2.py \
  --target_path $LOCAL/models/Alpamayo-R1-10B \
  --init_draft_path $LOCAL/runs/dflash_L4_lr1e-4_ep15_bs16_warm_sharon1/draft_final.pt \
  --target_outputs_dir $LOCAL/runs/target_coc_outputs \
  --val_uuids_file $SPLITS/val_uuids_v3.json \
  --test_uuids_file $SPLITS/test_uuids_v3.json \
  --output_dir $OUT \
  --num_epochs 2 --lr 1e-5 --kl_weight 0.02 \
  --k_samples 4 --k_chunk_size 4 --temperature 1.0 \
  --contamination_N 5 --accept_bonus 0.0 --block_start_decay 0.8 \
  --log_interval 25 --save_interval 250 --topk_save 8 \
  --ref_update_interval 1000 --ref_update_mode replace \
  --ref_update_gate none \
  --no_wandb \
  > $LOG 2>&1 &

echo "launched pid=$!  log=$LOG  outdir=$OUT"
sleep 1
ps -ef | grep "ref_update_interval 1000" | grep -v grep | head -3

#!/bin/bash
# Memory probe: launch K=20 RL training on 1 GPU for max_steps=5 to verify
# K-chunking keeps memory bounded.
set -e

LOCAL=/home/ubuntu/local_data
SCRIPTS=/home/ubuntu/katana_transfer/code/scripts
SPLITS=/home/ubuntu/katana_transfer/splits
OUT=$LOCAL/runs/rl_v2_N5_k20_PROBE
LOG=/tmp/rl_v2_N5_k20_PROBE.log
mkdir -p $OUT

source /home/ubuntu/alpamayo_env/bin/activate
export PYTHONPATH=/home/ubuntu/katana_transfer/code/src
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

nohup torchrun --nproc_per_node=1 --master_port=29641 \
  $SCRIPTS/train_dflash_rl_action_v2.py \
  --target_path $LOCAL/models/Alpamayo-R1-10B \
  --init_draft_path $LOCAL/runs/dflash_L4_lr1e-4_ep15_bs16_warm_sharon1/draft_final.pt \
  --target_outputs_dir $LOCAL/runs/target_coc_outputs \
  --val_uuids_file $SPLITS/val_uuids_v3.json \
  --test_uuids_file $SPLITS/test_uuids_v3.json \
  --output_dir $OUT \
  --num_epochs 1 --max_steps 5 --lr 1e-5 --kl_weight 0.02 \
  --k_samples 20 --k_chunk_size 10 --temperature 1.0 \
  --contamination_N 5 --accept_bonus 0.0 --block_start_decay 0.8 \
  --log_interval 1 --save_interval 1000 --topk_save 0 --no_wandb \
  > $LOG 2>&1 &
echo "probe pid=$! log=$LOG"
sleep 1
ps -ef | grep "train_dflash_rl_action_v2.py" | grep -v grep | head -3

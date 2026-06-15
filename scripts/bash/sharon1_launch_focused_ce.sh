#!/bin/bash
# Focused-CE fine-tune of warm SFT.
# Hard CE loss only at greedy-rejected positions, KL anchor at matched positions.
# 3 epochs, lr=1e-5, kl_weight=1.0. 4 GPUs.
set -e

LOCAL=/home/ubuntu/local_data
SCRIPTS=/home/ubuntu/katana_transfer/code/scripts
SPLITS=/home/ubuntu/katana_transfer/splits

OUT=$LOCAL/runs/dflash_L4_focused_ce_warm_init
LOG=/tmp/dflash_L4_focused_ce.log
mkdir -p $OUT

source /home/ubuntu/alpamayo_env/bin/activate
export PYTHONPATH=/home/ubuntu/katana_transfer/code/src
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup torchrun --nproc_per_node=4 --master_port=29644 \
  $SCRIPTS/train_dflash_focused_ce.py \
  --target_path $LOCAL/models/Alpamayo-R1-10B \
  --target_outputs_dir $LOCAL/runs/target_coc_outputs \
  --val_uuids_file $SPLITS/val_uuids_v3.json \
  --test_uuids_file $SPLITS/test_uuids_v3.json \
  --output_dir $OUT \
  --pretrained_draft $LOCAL/runs/dflash_L4_lr1e-4_ep15_bs16_warm_sharon1/draft_final.pt \
  --num_draft_layers 4 --block_size 16 \
  --batch_size 4 --grad_accum_steps 1 --num_workers 4 \
  --lr 1e-5 --num_epochs 3 \
  --log_interval 10 --val_interval 200 --save_interval 500 --val_batches 50 \
  --use_mrope_draft --overlapping_blocks --random_mask \
  --kl_weight 1.0 --seed 42 \
  --topk_save_val 3 \
  --no_wandb \
  > $LOG 2>&1 &

echo "launched pid=$!  log=$LOG  outdir=$OUT"
sleep 1
ps -ef | grep "train_dflash_focused_ce.py" | grep -v grep | head -3

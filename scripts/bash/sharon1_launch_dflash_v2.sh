#!/bin/bash
# Re-run dflash_L4_lr1e-4_ep15_bs16 with same hyperparameters but with topk_save_val=3
# (top-3 checkpoints by lowest val/weighted_ce). Output saved as ..._v2_sharon1.
set -e

LOCAL=/home/ubuntu/local_data
SCRIPTS=/home/ubuntu/katana_transfer/code/scripts
SPLITS=/home/ubuntu/katana_transfer/splits

OUTDIR=$LOCAL/runs/dflash_L4_lr1e-4_ep15_bs16_v2_sharon1
LOG=/tmp/dflash_v2_launch.log
mkdir -p "$OUTDIR"

PYBIN=/home/ubuntu/alpamayo_env/bin/python
TORCHRUN=/home/ubuntu/alpamayo_env/bin/torchrun
export PYTHONPATH=/home/ubuntu/dflash_code:/home/ubuntu/katana_transfer/code/src

nohup $TORCHRUN --nproc_per_node=4 --master_port=29630 \
  $SCRIPTS/train_dflash_distillation_v3.py \
  --target_path $LOCAL/models/Alpamayo-R1-10B \
  --target_outputs_dir $LOCAL/runs/target_coc_outputs \
  --val_uuids_file $SPLITS/val_uuids_v3.json \
  --test_uuids_file $SPLITS/test_uuids_v3.json \
  --output_dir $OUTDIR \
  --num_draft_layers 4 \
  --block_size 16 \
  --batch_size 4 \
  --grad_accum_steps 1 \
  --num_workers 4 \
  --lr 1e-4 \
  --num_epochs 15 \
  --log_interval 10 \
  --val_interval 200 \
  --save_interval 500 \
  --val_batches 50 \
  --use_mrope_draft \
  --overlapping_blocks \
  --random_mask \
  --kl_weight 1.0 \
  --seed 42 \
  --warm_start \
  --topk_save_val 3 \
  --wandb_project dflash-distillation \
  --wandb_run_name dflash_L4_lr1e-4_ep15_bs16_v2_sharon1 \
  > $LOG 2>&1 &

echo "launched pid=$!  log=$LOG  outdir=$OUTDIR"
sleep 1
ps -ef | grep "train_dflash_distillation_v3.py\|--master_port=29630" | grep -v grep | head -10

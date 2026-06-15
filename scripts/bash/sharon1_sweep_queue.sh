#!/bin/bash
# Sequential DFlash sweep experiments for sharon1.
# Run AFTER the current baseline (L1/bs4/lr1e-4) finishes.
#
# Usage: nohup bash sharon1_sweep_queue.sh > /tmp/sharon1_sweep.log 2>&1 &

set -e
NFS=/mnt/resv-harry-6f72s/dungda
SCRIPT=/home/ubuntu/alpamayo_code/scripts/train_dflash_distillation.py
COMMON="--target_path $NFS/models/Alpamayo-R1-10B \
    --ultrachat_dir $NFS/data/ultrachat_200k \
    --clips_dir $NFS/data/alpamayo_clips \
    --max_clips 5000 --max_ultrachat 50000 --num_epochs 3 \
    --wandb_project dflash-distillation \
    --save_via_tmp"

source /home/ubuntu/miniconda3/bin/activate alpamayo
export PYTHONPATH=/home/ubuntu/dflash_code:/home/ubuntu/alpamayo_code/src
export VLM_PATH=$NFS/models/Qwen3-VL-8B-Instruct
export PROCESSOR_PATH=$NFS/models/Qwen3-VL-2B-Instruct
export WANDB_API_KEY="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"

echo "=== sharon1 sweep queue started $(date) ==="

# Exp2: lr=5e-4
echo "--- Starting Exp2: L1/bs4/lr=5e-4 $(date) ---"
torchrun --nproc_per_node=4 $SCRIPT $COMMON \
    --output_dir $NFS/runs/sweep_lr5e4_v4 \
    --lr 5e-4 --num_draft_layers 1 --block_size 4 \
    --wandb_run_name sharon1_L1_bs4_lr5e4
echo "--- Exp2 done $(date) ---"

# Exp3: lr=1e-5
echo "--- Starting Exp3: L1/bs4/lr=1e-5 $(date) ---"
torchrun --nproc_per_node=4 $SCRIPT $COMMON \
    --output_dir $NFS/runs/sweep_lr1e5_v4 \
    --lr 1e-5 --num_draft_layers 1 --block_size 4 \
    --wandb_run_name sharon1_L1_bs4_lr1e5
echo "--- Exp3 done $(date) ---"

# Exp4: block_size=8
echo "--- Starting Exp4: L1/bs8/lr=1e-4 $(date) ---"
torchrun --nproc_per_node=4 $SCRIPT $COMMON \
    --output_dir $NFS/runs/sweep_bs8_v4 \
    --lr 1e-4 --num_draft_layers 1 --block_size 8 \
    --wandb_run_name sharon1_L1_bs8
echo "--- Exp4 done $(date) ---"

echo "=== sharon1 sweep queue finished $(date) ==="

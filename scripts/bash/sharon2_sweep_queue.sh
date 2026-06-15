#!/bin/bash
# Sequential DFlash sweep experiments for sharon2.
# Run AFTER the current L5/bs8 experiment finishes.
#
# Usage: nohup bash sharon2_sweep_queue.sh > /tmp/sharon2_sweep.log 2>&1 &

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

echo "=== sharon2 sweep queue started $(date) ==="

# Exp5: block_size=16
echo "--- Starting Exp5: L1/bs16/lr=1e-4 $(date) ---"
torchrun --nproc_per_node=4 $SCRIPT $COMMON \
    --output_dir $NFS/runs/sweep_bs16_v4 \
    --lr 1e-4 --num_draft_layers 1 --block_size 16 \
    --wandb_run_name sharon2_L1_bs16
echo "--- Exp5 done $(date) ---"

# Exp6: 2 draft layers
echo "--- Starting Exp6: L2/bs4/lr=1e-4 $(date) ---"
torchrun --nproc_per_node=4 $SCRIPT $COMMON \
    --output_dir $NFS/runs/sweep_2layers_v4 \
    --lr 1e-4 --num_draft_layers 2 --block_size 4 \
    --wandb_run_name sharon2_L2_bs4
echo "--- Exp6 done $(date) ---"

# Exp7: 5 draft layers, block_size=4
echo "--- Starting Exp7: L5/bs4/lr=1e-4 $(date) ---"
torchrun --nproc_per_node=4 $SCRIPT $COMMON \
    --output_dir $NFS/runs/sweep_5layers_v4 \
    --lr 1e-4 --num_draft_layers 5 --block_size 4 \
    --wandb_run_name sharon2_L5_bs4
echo "--- Exp7 done $(date) ---"

echo "=== sharon2 sweep queue finished $(date) ==="

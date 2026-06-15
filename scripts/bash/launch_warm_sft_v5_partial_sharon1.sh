#!/bin/bash
# Warm SFT v5 — DFlash L4 bs16 with --allow_partial_blocks.
# Same recipe as v4 (M-RoPE 3D fix) but ALSO allows blocks to extend past the
# end of the output (with masking). This recovers the ~66% of samples that
# v2/v4 silently skipped because num_gen < block_size + 1, and matches the
# sampling distribution where the chain proposes a full block_size regardless
# of remaining output. Runs on sharon1 GPUs 4-7.
set -e
source /home/ubuntu/alpamayo_env/bin/activate
export PYTHONPATH=/home/ubuntu/katana_transfer/code/src
export VLM_PATH=/home/ubuntu/local_data/models/Qwen3-VL-8B-Instruct
export PROCESSOR_PATH=/home/ubuntu/local_data/models/Qwen3-VL-2B-Instruct
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export HF_HOME=/home/ubuntu/.cache/huggingface
export CUDA_VISIBLE_DEVICES=4,5,6,7

DATA=/home/ubuntu/local_data
SPLITS=/home/ubuntu/katana_transfer/splits
RUN=dflash_L4_lr1e-4_ep15_bs16_warm_v5_partial_sharon1
OUTDIR=$DATA/runs/$RUN
mkdir -p $OUTDIR

cd /home/ubuntu/katana_transfer/code
torchrun --nproc_per_node=4 --master_port=29633 \
    scripts/train_dflash_distillation_v2.py \
    --target_path $DATA/models/Alpamayo-R1-10B \
    --target_outputs_dir $DATA/runs/target_coc_outputs \
    --val_uuids_file $SPLITS/val_uuids_v3.json \
    --test_uuids_file $SPLITS/test_uuids_v3.json \
    --output_dir $OUTDIR \
    --num_draft_layers 4 --block_size 16 \
    --batch_size 4 --grad_accum_steps 1 --num_workers 4 \
    --lr 1e-4 --num_epochs 15 \
    --log_interval 10 --val_interval 200 --save_interval 500 \
    --allow_partial_blocks \
    --overlapping_blocks --random_mask --kl_weight 1.0 \
    --warm_start --seed 42 --no_wandb
echo "DONE $RUN $(date)"

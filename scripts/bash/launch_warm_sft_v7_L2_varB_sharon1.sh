#!/bin/bash
# v7 L=2 variant B — 70% full mask + 30% discrete-mid {4,8,11}, position 1 ALWAYS masked.
# Runs on sharon1 GPUs 4-7 (4 GPUs).
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
RUN=dflash_L2_lr1e-4_ep50_bs16_warm_v7_varB_sharon1
OUTDIR=$DATA/runs/$RUN
mkdir -p $OUTDIR

cd /home/ubuntu/katana_transfer/code
torchrun --nproc_per_node=4 --master_port=29639 \
    scripts/train_dflash_distillation_v7.py \
    --target_path $DATA/models/Alpamayo-R1-10B \
    --target_outputs_dir $DATA/runs/target_coc_outputs \
    --val_uuids_file $SPLITS/val_uuids_v3.json \
    --test_uuids_file $SPLITS/test_uuids_v3.json \
    --output_dir $OUTDIR \
    --num_draft_layers 2 --block_size 16 \
    --num_target_features 5 \
    --warmup_ratio 0.04 \
    --batch_size 4 --grad_accum_steps 1 --num_workers 4 \
    --lr 1e-4 --num_epochs 50 \
    --log_interval 10 --val_interval 500 --save_interval 5000 \
    --allow_partial_blocks --overlapping_blocks \
    --full_mask_prob 0.7 --discrete_levels 4 8 11 --always_mask_pos1 \
    --kl_weight 1.0 \
    --warm_start --seed 42 --no_wandb
echo "DONE_V7_L2_varB $RUN $(date)"

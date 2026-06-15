#!/bin/bash
# Warm SFT v6 — DFlash L=2 (thinner draft) on sharon1 GPUs 4-7.
# Same recipe as v6 L4: num_target_features=5, warmup_ratio=0.04, random_mask
# off, allow_partial_blocks, overlapping_blocks. Only --num_draft_layers
# differs (2 vs 4). With features now decoupled from depth (paper recipe),
# this isolates whether thinner draft remains competitive.
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
RUN=dflash_L2_lr1e-4_ep15_bs16_warm_v6_sharon1
OUTDIR=$DATA/runs/$RUN
mkdir -p $OUTDIR

cd /home/ubuntu/katana_transfer/code
torchrun --nproc_per_node=4 --master_port=29635 \
    scripts/train_dflash_distillation_v6.py \
    --target_path $DATA/models/Alpamayo-R1-10B \
    --target_outputs_dir $DATA/runs/target_coc_outputs \
    --val_uuids_file $SPLITS/val_uuids_v3.json \
    --test_uuids_file $SPLITS/test_uuids_v3.json \
    --output_dir $OUTDIR \
    --num_draft_layers 2 --block_size 16 \
    --num_target_features 5 \
    --warmup_ratio 0.04 \
    --batch_size 4 --grad_accum_steps 1 --num_workers 4 \
    --lr 1e-4 --num_epochs 15 \
    --log_interval 10 --val_interval 200 --save_interval 500 \
    --allow_partial_blocks \
    --overlapping_blocks --kl_weight 1.0 \
    --warm_start --seed 42 --no_wandb
echo "DONE_V6_L2 $RUN $(date)"

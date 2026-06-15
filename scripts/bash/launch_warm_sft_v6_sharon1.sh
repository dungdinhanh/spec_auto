#!/bin/bash
# Warm SFT v6 — DFlash L4 bs16 with three paper-faithful tweaks on top of v5_partial:
#   1. --num_target_features 5  (was tied to num_draft_layers=4 in v2/v5)
#   2. --warmup_ratio 0.04       (was 0, no warmup in v2/v5)
#   3. random_mask DISABLED      (paper masks all B-1 positions after anchor)
# Keeps --allow_partial_blocks and --overlapping_blocks (v5_partial behaviour).
# Runs on sharon1 GPUs 0-3 (FlatRoPE = no use_mrope*_draft flags).
set -e
source /home/ubuntu/alpamayo_env/bin/activate
export PYTHONPATH=/home/ubuntu/katana_transfer/code/src
export VLM_PATH=/home/ubuntu/local_data/models/Qwen3-VL-8B-Instruct
export PROCESSOR_PATH=/home/ubuntu/local_data/models/Qwen3-VL-2B-Instruct
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export HF_HOME=/home/ubuntu/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0,1,2,3

DATA=/home/ubuntu/local_data
SPLITS=/home/ubuntu/katana_transfer/splits
RUN=dflash_L4_lr1e-4_ep15_bs16_warm_v6_sharon1
OUTDIR=$DATA/runs/$RUN
mkdir -p $OUTDIR

cd /home/ubuntu/katana_transfer/code
torchrun --nproc_per_node=4 --master_port=29634 \
    scripts/train_dflash_distillation_v6.py \
    --target_path $DATA/models/Alpamayo-R1-10B \
    --target_outputs_dir $DATA/runs/target_coc_outputs \
    --val_uuids_file $SPLITS/val_uuids_v3.json \
    --test_uuids_file $SPLITS/test_uuids_v3.json \
    --output_dir $OUTDIR \
    --num_draft_layers 4 --block_size 16 \
    --num_target_features 5 \
    --warmup_ratio 0.04 \
    --batch_size 4 --grad_accum_steps 1 --num_workers 4 \
    --lr 1e-4 --num_epochs 15 \
    --log_interval 10 --val_interval 200 --save_interval 500 \
    --allow_partial_blocks \
    --overlapping_blocks --kl_weight 1.0 \
    --warm_start --seed 42 --no_wandb
echo "DONE_V6 $RUN $(date)"

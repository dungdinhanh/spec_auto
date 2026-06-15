#!/bin/bash
# More-data probe — ARM B (+data): v6-RM recipe, CE-only (kl_weight=0),
# trained on combined ~14.6k (old + 3000 new). L=2, ep20.
# sharon1 GPUs 4-7.
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
RUN=dflash_L2_probe_B_combined_ep20_ceonly_sharon1
OUTDIR=$DATA/runs/$RUN
mkdir -p $OUTDIR

cd /home/ubuntu/katana_transfer/code
torchrun --nproc_per_node=4 --master_port=29639 \
    scripts/train_dflash_distillation_v7.py \
    --target_path $DATA/models/Alpamayo-R1-10B \
    --target_outputs_dir $DATA/runs/target_coc_outputs_combined \
    --val_uuids_file $SPLITS/val_uuids_v3.json \
    --test_uuids_file $SPLITS/test_uuids_v3.json \
    --output_dir $OUTDIR \
    --num_draft_layers 2 --block_size 16 --num_target_features 5 \
    --warmup_ratio 0.04 \
    --batch_size 4 --grad_accum_steps 1 --num_workers 4 \
    --lr 1e-4 --num_epochs 20 --max_clips 16000 \
    --log_interval 10 --val_interval 500 --save_interval 5000 \
    --allow_partial_blocks --overlapping_blocks --random_mask \
    --kl_weight 0 \
    --warm_start --seed 42 --no_wandb
echo "DONE_PROBE_B $RUN $(date)"

#!/bin/bash
#PBS -q R7936711
#PBS -l select=1:ncpus=48:mem=512gb:ngpus=8:gpu_model=H200
#PBS -l walltime=72:00:00
#PBS -j oe

# Paths
SCRATCH=/srv/scratch/cruise/dungda/path_a
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_API_KEY="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"
export HF_HOME=$SCRATCH/cache/huggingface
export VLM_PATH=$SCRATCH/models/Qwen3-VL-8B-Instruct
export PROCESSOR_PATH=$SCRATCH/models/Qwen3-VL-2B-Instruct

# Activate env
source $SCRATCH/envs/alpamayo/bin/activate
export PYTHONPATH=$SCRATCH/code/dflash:$SCRATCH/code/alpamayo_repo/src:$PYTHONPATH

cd $SCRATCH/code/alpamayo_repo

# Run with 8 GPUs via torchrun
torchrun --nproc_per_node=8 scripts/train_dflash_distillation.py \
    --target_path $SCRATCH/models/Alpamayo-R1-10B \
    --clips_dir $SCRATCH/data/alpamayo_clips \
    --ultrachat_dir $SCRATCH/data/ultrachat_200k \
    --output_dir ${OUTPUT_DIR:-$SCRATCH/runs/dflash_distill} \
    --max_clips ${MAX_CLIPS:-5000} \
    --max_ultrachat ${MAX_ULTRACHAT:-50000} \
    --num_epochs ${NUM_EPOCHS:-3} \
    --lr ${LR:-1e-4} \
    --grad_accum_steps ${GRAD_ACCUM:-4} \
    --num_draft_layers ${NUM_LAYERS:-1} \
    --block_size ${BLOCK_SIZE:-4} \
    --log_interval 10 \
    --save_interval 500

echo "=== DONE ==="

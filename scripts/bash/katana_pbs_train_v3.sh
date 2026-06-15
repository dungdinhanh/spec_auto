#!/bin/bash
#PBS -q R7936711
#PBS -l select=1:ncpus=48:mem=512gb:ngpus=4:gpu_model=H200
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -o /srv/scratch/z3552416/logs/dflash_train_v3.log
#PBS -N dflash_v3

set -e

export SCRATCH=/srv/scratch/z3552416
export FLORA=/srv/scratch/flora/dungda
export HF_HOME=$SCRATCH/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3
export WANDB_API_KEY="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"

# Point alpamayo base_model/helper at local Qwen3-VL backbone + processor
export VLM_PATH=$FLORA/models/Qwen3-VL-8B-Instruct
export PROCESSOR_PATH=$FLORA/models/Qwen3-VL-2B-Instruct

source $SCRATCH/envs/alpamayo/bin/activate

cd $PBS_O_WORKDIR
export PYTHONPATH=$PBS_O_WORKDIR/src:$PYTHONPATH

RUN_NAME=${RUN_NAME:-exp_L${NUM_LAYERS:-3}_mrope_bs${BATCH_SIZE:-4}_v3}
OUTPUT_DIR=$FLORA/runs/$RUN_NAME
mkdir -p $OUTPUT_DIR

echo "=== DFlash v3 training start $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "output_dir=$OUTPUT_DIR"

JOB_NUM=$(echo $PBS_JOBID | sed 's/\..*//' | grep -oE '[0-9]+')
MASTER_PORT=$(( 29500 + (JOB_NUM % 1000) ))
echo "master_port=$MASTER_PORT job=$PBS_JOBID"

torchrun --nproc_per_node=4 --master_port=$MASTER_PORT scripts/train_dflash_distillation_v2.py \
    --target_path $FLORA/models/Alpamayo-R1-10B \
    --target_outputs_dir $FLORA/runs/target_coc_outputs \
    --val_uuids_file $FLORA/runs/splits/val_uuids_v3.json \
    --test_uuids_file $FLORA/runs/splits/test_uuids_v3.json \
    --output_dir $OUTPUT_DIR \
    --num_draft_layers ${NUM_LAYERS:-3} \
    --block_size ${BLOCK_SIZE:-8} \
    --batch_size ${BATCH_SIZE:-4} \
    --grad_accum_steps ${GRAD_ACCUM:-1} \
    --num_workers ${NUM_WORKERS:-4} \
    --lr ${LR:-1e-4} \
    --num_epochs ${NUM_EPOCHS:-3} \
    --log_interval 10 \
    --val_interval 200 \
    --save_interval 500 \
    --use_mrope_draft \
    --overlapping_blocks \
    --random_mask \
    --kl_weight ${KL_WEIGHT:-1.0} \
    --seed 42 \
    --wandb_project dflash-distillation \
    --wandb_run_name $RUN_NAME

echo "=== DFlash v3 training done $(date) ==="

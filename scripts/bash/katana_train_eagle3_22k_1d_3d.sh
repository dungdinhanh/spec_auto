#!/bin/bash
#PBS -q R7936711
#PBS -l select=1:host=k201:ncpus=48:ngpus=8:mem=480gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -N eagle3_22k_1d3d
# Full-22k EAGLE-3 training (paper recipe) — BOTH 1D and 3D variants in parallel.
# 1D on GPUs 0-3, 3D on GPUs 4-7. ep20, batch 2 × accum 2, lr 1e-4.
set -e

SCRATCH=/srv/scratch/cruise/dungda/path_a
export VLM_PATH=$SCRATCH/models/Qwen3-VL-8B-Instruct
export PROCESSOR_PATH=$SCRATCH/models/Qwen3-VL-2B-Instruct
export HF_HOME=$SCRATCH/cache/huggingface
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
source $SCRATCH/envs/alpamayo/bin/activate
export PYTHONPATH=$SCRATCH/code/dflash:$SCRATCH/code/alpamayo_repo/src:$PYTHONPATH
cd $SCRATCH/code/alpamayo_repo

T=$SCRATCH/models/Alpamayo-R1-10B
DATA=$SCRATCH/runs/target_coc_outputs_all
V=$SCRATCH/code/splits_v3/val_uuids_v3.json
TE=$SCRATCH/code/splits_v3/test_uuids_v3.json

common_args() {
  echo "--target_path $T --target_outputs_dir $DATA \
    --val_uuids_file $V --test_uuids_file $TE \
    --target_layer_ids 1,17,32 --rollout_length 7 \
    --num_epochs 20 --lr 1e-4 \
    --batch_size 2 --grad_accum_steps 2 --num_workers 2 \
    --log_interval 25 --val_interval 1000 --save_interval 100000 \
    --no_wandb"
}

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29671 \
    scripts/train_eagle3.py \
    --output_dir $SCRATCH/runs/eagle3_22k_1d_ep20_katana \
    $(common_args) > $SCRATCH/runs/eagle3_22k_1d.log 2>&1 &
P1=$!

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29672 \
    scripts/train_eagle3.py \
    --output_dir $SCRATCH/runs/eagle3_22k_3d_ep20_katana \
    --use_mrope3d_draft \
    $(common_args) > $SCRATCH/runs/eagle3_22k_3d.log 2>&1 &
P3=$!

echo "Launched EAGLE-3 1D(pid=$P1, GPU0-3) 3D(pid=$P3, GPU4-7) at $(date)"
wait $P1; echo "1D exit=$?"
wait $P3; echo "3D exit=$?"
echo "DONE_EAGLE3_22k_1D3D $(date)"

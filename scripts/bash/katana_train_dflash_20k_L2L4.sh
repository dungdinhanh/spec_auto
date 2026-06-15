#!/bin/bash
#PBS -q R7936711
#PBS -l select=1:host=k201:ncpus=48:ngpus=8:mem=480gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -N dflash20k_L2L4
# Full-20k DFlash training on the H200 reservation node (k201).
# ONE job, 8 GPUs: L=2 on GPUs 0-3, L=4 on GPUs 4-7, in parallel.
# CE-only (new 10k clips have no cached logits); v6-RM recipe otherwise.
# Same held-out val/test UUID split as before.
set -e

SCRATCH=/srv/scratch/cruise/dungda/path_a
export VLM_PATH=$SCRATCH/models/Qwen3-VL-8B-Instruct
export PROCESSOR_PATH=$SCRATCH/models/Qwen3-VL-2B-Instruct
export HF_HOME=$SCRATCH/cache/huggingface
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
source $SCRATCH/envs/alpamayo/bin/activate
export PYTHONPATH=$SCRATCH/code/dflash:$SCRATCH/code/alpamayo_repo/src:$PYTHONPATH
cd $SCRATCH/code/alpamayo_repo

T=$SCRATCH/models/Alpamayo-R1-10B
DATA=$SCRATCH/runs/target_coc_outputs_all          # combined 22k (symlinks)
V=$SCRATCH/code/splits_v3/val_uuids_v3.json
TE=$SCRATCH/code/splits_v3/test_uuids_v3.json

common_args() {
  echo "--target_path $T --target_outputs_dir $DATA \
    --val_uuids_file $V --test_uuids_file $TE \
    --block_size 16 --num_target_features 5 --warmup_ratio 0.04 \
    --batch_size 4 --grad_accum_steps 1 --num_workers 6 \
    --lr 1e-4 --num_epochs 30 --max_clips 30000 \
    --log_interval 20 --val_interval 1000 --save_interval 100000 \
    --allow_partial_blocks --overlapping_blocks --random_mask \
    --kl_weight 0 --warm_start --seed 42 --no_wandb"
}

# L=2 on GPUs 0-3
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29651 \
    scripts/train_dflash_distillation_v7.py \
    --num_draft_layers 2 \
    --output_dir $SCRATCH/runs/dflash_L2_20k_ep30_ceonly_katana \
    $(common_args) > $SCRATCH/runs/dflash_20k_L2.log 2>&1 &
P2=$!

# L=4 on GPUs 4-7
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29652 \
    scripts/train_dflash_distillation_v7.py \
    --num_draft_layers 4 \
    --output_dir $SCRATCH/runs/dflash_L4_20k_ep30_ceonly_katana \
    $(common_args) > $SCRATCH/runs/dflash_20k_L4.log 2>&1 &
P4=$!

echo "Launched L2(pid=$P2) on GPU0-3, L4(pid=$P4) on GPU4-7 at $(date)"
wait $P2; echo "L2 exit=$?"
wait $P4; echo "L4 exit=$?"
echo "DONE_DFLASH_20k_L2L4 $(date)"

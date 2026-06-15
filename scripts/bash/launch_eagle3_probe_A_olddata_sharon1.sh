#!/bin/bash
# EAGLE-3 more-data probe — ARM A (baseline): trained on EXISTING ~11.2k target
# outputs only. Same paper recipe as eagle3_paper_v2 (layer_ids 1,17,32,
# rollout 7, live soft-KL). ep20. sharon1 GPUs 0-3.
set -e
source /home/ubuntu/alpamayo_env/bin/activate
export PYTHONPATH=/home/ubuntu/katana_transfer/code/src
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

LOCAL=/home/ubuntu/local_data
T=$LOCAL/models/Alpamayo-R1-10B
C=$LOCAL/runs/target_coc_outputs
V=/home/ubuntu/katana_transfer/splits/val_uuids_v3.json
TE=/home/ubuntu/katana_transfer/splits/test_uuids_v3.json
OUT=$LOCAL/runs/eagle3_probe_A_olddata_ep20_sharon1
mkdir -p $OUT

torchrun --nproc_per_node=4 --master_port=29642 \
    /home/ubuntu/katana_transfer/code/scripts/train_eagle3.py \
    --target_path $T --target_outputs_dir $C \
    --val_uuids_file $V --test_uuids_file $TE \
    --output_dir $OUT \
    --target_layer_ids 1,17,32 --rollout_length 7 \
    --num_epochs 20 --lr 1e-4 \
    --batch_size 2 --grad_accum_steps 2 --num_workers 2 \
    --log_interval 25 --val_interval 500 --save_interval 100000 \
    --no_wandb
echo "DONE_EAGLE3_PROBE_A $(date)"

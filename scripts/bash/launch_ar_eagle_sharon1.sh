#!/bin/bash
# EAGLE-3-style AR draft baseline. 4-GPU torchrun on sharon1 GPUs 0-3.
# Same data + warm-start convention as the DFlash draft for fair comparison.
# 15 epochs, lr=1e-4, batch=4 per GPU (effective batch 16).
source /home/ubuntu/alpamayo_env/bin/activate
export PYTHONPATH=/home/ubuntu/katana_transfer/code/src
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
T=/home/ubuntu/local_data/models/Alpamayo-R1-10B
C=/home/ubuntu/local_data/runs/target_coc_outputs
V=/home/ubuntu/katana_transfer/splits/val_uuids_v3.json
TE=/home/ubuntu/katana_transfer/splits/test_uuids_v3.json
torchrun --nproc_per_node=4 --master_port=29630 \
    /home/ubuntu/katana_transfer/code/scripts/train_ar_draft.py \
    --target_path $T \
    --target_outputs_dir $C \
    --val_uuids_file $V --test_uuids_file $TE \
    --output_dir /home/ubuntu/local_data/runs/ar_eagle_L4_v3_sharon1 \
    --num_draft_layers 4 \
    --target_layer_idx -1 \
    --num_epochs 15 --lr 1e-4 \
    --batch_size 4 --grad_accum_steps 1 --num_workers 2 \
    --warm_start \
    --log_interval 25 --val_interval 500 --save_interval 1000 \
    --no_wandb
echo "DONE ar_eagle_L4_v3_sharon1 $(date)"

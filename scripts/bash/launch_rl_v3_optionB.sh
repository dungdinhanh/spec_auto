#!/bin/bash
# v3 Option B: w_cons>0 — rule-based meta-action consistency added.
# Reward = w_traj·(-MSE) + w_cons·1[meta_long & meta_lat match] + w_text·token_overlap.
source /home/ubuntu/miniconda3/bin/activate alpamayo
export PYTHONPATH=/home/ubuntu/dflash_code:/home/ubuntu/alpamayo_code/src
export VLM_PATH=/mnt/resv-harry-6f72s/dungda/models/Qwen3-VL-8B-Instruct
export PROCESSOR_PATH=/mnt/resv-harry-6f72s/dungda/models/Qwen3-VL-2B-Instruct
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
cd /home/ubuntu/alpamayo_code
exec torchrun --nproc_per_node=4 --master_port=29612 \
    /home/ubuntu/alpamayo_code/scripts/train_dflash_rl_action_v3.py \
    --target_path /mnt/resv-harry-6f72s/dungda/models/Alpamayo-R1-10B \
    --init_draft_path /mnt/resv-harry-6f72s/dungda/runs/rl_init/draft_final.pt \
    --target_outputs_dir /mnt/resv-harry-6f72s/dungda/runs/target_coc_outputs \
    --val_uuids_file /mnt/resv-harry-6f72s/dungda/runs/splits/val_uuids_v3.json \
    --test_uuids_file /mnt/resv-harry-6f72s/dungda/runs/splits/test_uuids_v3.json \
    --output_dir /mnt/resv-harry-6f72s/dungda/runs/rl_v3_optionB_K5 \
    --num_epochs 1 --lr 1e-6 --kl_weight 0.02 \
    --k_samples 5 --k_chunk_size 5 --temperature 1.0 \
    --w_traj 1.0 --w_cons 0.1 --w_text 0.5 \
    --enable_r_cons \
    --consistency_horizon 16 --eps_long 0.05 --eps_lat 0.10 \
    --contamination_N 3 \
    --log_interval 5 --save_interval 500 \
    --wandb_project dflash-rl-action \
    --wandb_run_name rl_v3_optionB_K5_w_traj1_w_cons0p1_w_text0p5

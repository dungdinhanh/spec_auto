#!/bin/bash
# v4 AARL with --filter_to_rejection_blocks on, GPUs 0-3, lr=1e-6 (matches v3
# baseline lr). Direct comparison vs v3 AARL static-ref anchor (val 4.544,
# test 4.225). Same warm-SFT init, same data, same K/N/anchor settings.
source /home/ubuntu/alpamayo_env/bin/activate
export PYTHONPATH=/home/ubuntu/katana_transfer/code/src
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
T=/home/ubuntu/local_data/models/Alpamayo-R1-10B
I=/home/ubuntu/local_data/runs/dflash_L4_lr1e-4_ep15_bs16_warm_v5_partial_sharon1/draft_final.pt
C=/home/ubuntu/local_data/runs/target_coc_outputs
V=/home/ubuntu/katana_transfer/splits/val_uuids_v3.json
TE=/home/ubuntu/katana_transfer/splits/test_uuids_v3.json
torchrun --nproc_per_node=4 --master_port=29611 \
    /home/ubuntu/katana_transfer/code/scripts/train_dflash_rl_action_v4.py \
    --target_path $T --init_draft_path $I \
    --target_outputs_dir $C --val_uuids_file $V --test_uuids_file $TE \
    --output_dir /home/ubuntu/local_data/runs/aarl_v4_filter_lr1e-6 \
    --num_epochs 2 --lr 1e-6 --kl_weight 0.02 \
    --k_samples 5 --k_chunk_size 5 --temperature 1.0 \
    --w_traj 1.0 --w_cons 0.0 --w_text 0.5 \
    --contamination_N 3 --anchor_source ref \
    --filter_to_rejection_blocks \
    --log_interval 5 --save_interval 500 --no_wandb
echo "V4_FILTER_LR1E6_DONE $(date)"

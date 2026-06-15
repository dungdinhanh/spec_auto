#!/bin/bash
# v3 Option A with --anchor_source ref, K=10, k_chunk_size=5.
# Same as winning config except K varies. lr=1e-6, N=3, 2 epochs.
source /home/ubuntu/alpamayo_env/bin/activate
export PYTHONPATH=/home/ubuntu/katana_transfer/code/src
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
T=/home/ubuntu/local_data/models/Alpamayo-R1-10B
I=/home/ubuntu/local_data/runs/dflash_L4_lr1e-4_ep15_bs16_warm_sharon1/draft_final.pt
C=/home/ubuntu/local_data/runs/target_coc_outputs
V=/home/ubuntu/katana_transfer/splits/val_uuids_v3.json
TE=/home/ubuntu/katana_transfer/splits/test_uuids_v3.json
torchrun --nproc_per_node=4 --master_port=29625 \
    /home/ubuntu/katana_transfer/code/scripts/train_dflash_rl_action_v3.py \
    --target_path $T --init_draft_path $I \
    --target_outputs_dir $C --val_uuids_file $V --test_uuids_file $TE \
    --output_dir /home/ubuntu/local_data/runs/rl_v3_optA_ref_K10 \
    --num_epochs 2 --lr 1e-6 --kl_weight 0.02 \
    --k_samples 10 --k_chunk_size 5 --temperature 1.0 \
    --w_traj 1.0 --w_cons 0.0 --w_text 0.5 \
    --contamination_N 3 \
    --anchor_source ref \
    --log_interval 5 --save_interval 500 --no_wandb
echo "DONE rl_v3_optA_ref_K10 $(date)"

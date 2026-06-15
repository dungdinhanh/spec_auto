#!/bin/bash
# AARL v4 on top of best L=2 14k SFT init (probe B ep15, off-L=5.152, S=4.42).
# K=32 (vs prior K=5) — bigger group for lower-variance GRPO baseline.
# 8-GPU sharon1, ep2 (matches prior K=5 recipe for parameter-update parity).
set -e
source /home/ubuntu/alpamayo_env/bin/activate
export PYTHONPATH=/home/ubuntu/katana_transfer/code/src
export VLM_PATH=/home/ubuntu/local_data/models/Qwen3-VL-8B-Instruct
export PROCESSOR_PATH=/home/ubuntu/local_data/models/Qwen3-VL-2B-Instruct
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

DATA=/home/ubuntu/local_data
T=$DATA/models/Alpamayo-R1-10B
I=$DATA/runs/dflash_L2_probe_B_combined_ep20_ceonly_sharon1/draft_epoch_15.pt
C=$DATA/runs/target_coc_outputs_combined          # 14,641 clips (old 11.6k stripped + new 3k)
V=/home/ubuntu/katana_transfer/splits/val_uuids_v3.json
TE=/home/ubuntu/katana_transfer/splits/test_uuids_v3.json
OUT=$DATA/runs/aarl_L2_14k_K32_ep2_sharon1
mkdir -p $OUT

cd /home/ubuntu/katana_transfer/code
torchrun --nproc_per_node=8 --master_port=29680 \
    scripts/train_dflash_rl_action_v4.py \
    --target_path $T --init_draft_path $I \
    --target_outputs_dir $C --val_uuids_file $V --test_uuids_file $TE \
    --output_dir $OUT \
    --num_target_features 5 \
    --num_epochs 2 --lr 1e-6 --kl_weight 0.02 \
    --k_samples 32 --k_chunk_size 4 --temperature 1.0 \
    --w_traj 1.0 --w_cons 0.0 --w_text 0.5 \
    --contamination_N 3 \
    --anchor_source ref \
    --filter_to_rejection_blocks \
    --log_interval 5 --save_interval 500 --no_wandb
echo "DONE_AARL_L2_14k_K32 $(date)"

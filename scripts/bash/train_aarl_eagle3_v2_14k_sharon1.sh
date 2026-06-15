#!/bin/bash
# EAGLE-3 AARL v2 on 14k_combined data — sharon1 (8 H100 NVL).
# Init from the 22k EAGLE-3 1D ckpt (best EAGLE-3 SFT we have).
# Uses J5/E1 recipe: N=5, max=20, lr=1e-6, w_traj=1.0, w_text=0.5, kl=0.02.
set -e
source /home/ubuntu/alpamayo_env/bin/activate
export PYTHONPATH=/home/ubuntu/katana_transfer/code/src
export VLM_PATH=/home/ubuntu/local_data/models/Qwen3-VL-8B-Instruct
export PROCESSOR_PATH=/home/ubuntu/local_data/models/Qwen3-VL-2B-Instruct
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

DATA=/home/ubuntu/local_data
T=$DATA/models/Alpamayo-R1-10B
I=$DATA/runs/eagle3_22k_1d_ep20_katana/draft_final.pt
C=$DATA/runs/target_coc_outputs_combined
V=/home/ubuntu/katana_transfer/splits/val_uuids_v3.json
TE=/home/ubuntu/katana_transfer/splits/test_uuids_v3.json
OUT=$DATA/runs/aarl_eagle3_v2_14k_sharon1_N5_max20
mkdir -p $OUT

cd /home/ubuntu/katana_transfer/code
torchrun --nproc_per_node=8 --master_port=29760 \
    scripts/train_eagle3_rl_action_v2.py \
    --target_path $T --init_draft_path $I \
    --target_outputs_dir $C --val_uuids_file $V --test_uuids_file $TE \
    --output_dir $OUT \
    --block_size 8 \
    --num_epochs 2 --lr 1e-6 --kl_weight 0.02 \
    --k_samples 32 --k_chunk_size 4 --temperature 1.0 \
    --w_traj 1.0 --w_cons 0.0 --w_text 0.5 \
    --multiblock_N 5 --multiblock_max_total 20 \
    --anchor_source ref \
    --filter_to_rejection_blocks \
    --log_interval 5 --save_interval 500 --no_wandb
echo "DONE_AARL_EAGLE3_14k_SHARON1 $(date)"

#!/bin/bash
# EAGLE-3 paper-faithful baseline with 3D M-RoPE draft.
# Same training recipe as v2 (1D, our contribution) but with `--use_mrope3d_draft`
# so the draft's rotary mirrors target's M-RoPE 3D — the natural baseline for
# multimodal targets per project_eagle3_1d_vs_3d_claim.md.
# Runs on sharon1 GPUs 0-3 (4-7 reserved for warm SFT v5 partial-blocks).
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
OUT=$LOCAL/runs/eagle3_paper_v3_3dmrope_sharon1
mkdir -p $OUT

torchrun --nproc_per_node=4 --master_port=29634 \
    /home/ubuntu/katana_transfer/code/scripts/train_eagle3.py \
    --target_path $T --target_outputs_dir $C \
    --val_uuids_file $V --test_uuids_file $TE \
    --output_dir $OUT \
    --target_layer_ids 1,17,32 --rollout_length 7 \
    --use_mrope3d_draft \
    --num_epochs 40 --lr 1e-4 \
    --batch_size 2 --grad_accum_steps 2 --num_workers 2 \
    --log_interval 25 --val_interval 500 --save_interval 1000 \
    --no_wandb
echo "DONE eagle3_paper_v3_3dmrope_sharon1 $(date)"

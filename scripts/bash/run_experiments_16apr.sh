#!/bin/bash
# DFlash draft experiments — April 16, 2026
# Run on sharon2 with 4×H100 NVL
#
# Experiment plan:
#   Phase 1: Overlapping+noise vs baselines (2 layers, lr=1e-4)
#   Phase 2: Layer sweep (1,2,3,4 layers) with best settings from phase 1
#   Phase 3: LR sweep (1e-4, 5e-4, 1e-3) with best layer count
#
# All use: block_size=8, random init, 3 epochs, 8000 train clips

set -e

source /home/ubuntu/miniconda3/bin/activate alpamayo
export PYTHONPATH=/home/ubuntu/dflash_code:/home/ubuntu/alpamayo_code/src
export VLM_PATH=/mnt/resv-harry-6f72s/dungda/models/Qwen3-VL-8B-Instruct
export PROCESSOR_PATH=/mnt/resv-harry-6f72s/dungda/models/Qwen3-VL-2B-Instruct
cd /home/ubuntu/alpamayo_code

TARGET=/mnt/resv-harry-6f72s/dungda/models/Alpamayo-R1-10B
TARGET_OUTPUTS=/mnt/resv-harry-6f72s/dungda/runs/target_coc_outputs
CLIPS=/mnt/resv-harry-6f72s/dungda/data/alpamayo_clips
RUN_BASE=/mnt/resv-harry-6f72s/dungda/runs
RESULTS_FILE=/tmp/experiment_results_16apr.txt

> $RESULTS_FILE

run_experiment() {
    local NAME=$1
    local LAYERS=$2
    local LR=$3
    local EXTRA_FLAGS=$4
    local OUT_DIR=$RUN_BASE/exp_${NAME}

    echo "========================================" | tee -a $RESULTS_FILE
    echo "EXPERIMENT: $NAME (layers=$LAYERS, lr=$LR, flags=$EXTRA_FLAGS)" | tee -a $RESULTS_FILE
    echo "========================================" | tee -a $RESULTS_FILE

    # Train
    echo "[$(date)] Training..." | tee -a $RESULTS_FILE
    torchrun --nproc_per_node=4 scripts/train_dflash_distillation.py \
        --target_path $TARGET \
        --target_outputs_dir $TARGET_OUTPUTS \
        --output_dir $OUT_DIR \
        --max_clips 8000 --val_clips 1000 --num_epochs 3 \
        --lr $LR --num_draft_layers $LAYERS --block_size 8 \
        --kl_weight 1.0 --grad_accum_steps 4 --save_interval 500 \
        --seed 42 --no_wandb \
        $EXTRA_FLAGS \
        > /tmp/exp_${NAME}.log 2>&1

    echo "[$(date)] Training done." | tee -a $RESULTS_FILE

    # Extract final loss
    FINAL_LOSS=$(grep "step 2000/2000" /tmp/exp_${NAME}.log | tail -1)
    echo "Final loss: $FINAL_LOSS" | tee -a $RESULTS_FILE

    # Test accuracy on output region (50 clips)
    echo "[$(date)] Testing..." | tee -a $RESULTS_FILE
    CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_draft_output_region.py \
        --target_path $TARGET \
        --draft_path $OUT_DIR/draft_final.pt \
        --target_outputs_dir $TARGET_OUTPUTS \
        --num_draft_layers $LAYERS \
        --block_size 8 \
        --num_clips 50 \
        2>&1 | tee -a $RESULTS_FILE

    echo "" >> $RESULTS_FILE
}

# ============================================================
# Phase 1: Overlapping + noise vs baselines (2 layers, lr=1e-4)
# ============================================================
echo "=== PHASE 1: Block strategy comparison (2 layers, lr=1e-4) ===" | tee -a $RESULTS_FILE

run_experiment "L2_nonoverlap_fullmask" 2 1e-4 ""
run_experiment "L2_overlap_fullmask" 2 1e-4 "--overlapping_blocks"
run_experiment "L2_overlap_randmask" 2 1e-4 "--overlapping_blocks --random_mask"
run_experiment "L2_nonoverlap_randmask" 2 1e-4 "--random_mask"

# ============================================================
# Phase 2: Layer sweep with best settings from phase 1
# (Assume overlapping+randmask is best; run all anyway)
# ============================================================
echo "=== PHASE 2: Layer sweep (overlap+randmask, lr=1e-4) ===" | tee -a $RESULTS_FILE

run_experiment "L1_overlap_randmask" 1 1e-4 "--overlapping_blocks --random_mask"
# L2 already done in phase 1
run_experiment "L3_overlap_randmask" 3 1e-4 "--overlapping_blocks --random_mask"
run_experiment "L4_overlap_randmask" 4 1e-4 "--overlapping_blocks --random_mask"

# ============================================================
# Phase 3: LR sweep with 2 layers (most likely sweet spot)
# ============================================================
echo "=== PHASE 3: LR sweep (2 layers, overlap+randmask) ===" | tee -a $RESULTS_FILE

# lr=1e-4 already done in phase 1
run_experiment "L2_overlap_randmask_lr5e4" 2 5e-4 "--overlapping_blocks --random_mask"
run_experiment "L2_overlap_randmask_lr1e3" 2 1e-3 "--overlapping_blocks --random_mask"

echo "=== ALL EXPERIMENTS DONE $(date) ===" | tee -a $RESULTS_FILE
echo "Results at: $RESULTS_FILE"

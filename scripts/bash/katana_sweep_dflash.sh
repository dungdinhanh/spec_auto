#!/bin/bash
# Submit DFlash block diffusion distillation sweep on Katana reservation.
#
# Usage:
#   bash scripts/katana_sweep_dflash.sh

set -e
SCRATCH=/srv/scratch/cruise/dungda/path_a
JOB_SCRIPT=$SCRATCH/code/alpamayo_repo/scripts/katana_train_dflash.sh

mkdir -p $SCRATCH/runs/sweep_logs

echo "=== DFlash Block Diffusion Distillation Sweep ==="

# --- Experiment 1: Baseline (1 layer, block_size=4, lr=1e-4) ---
JID1=$(qsub -v "OUTPUT_DIR=$SCRATCH/runs/sweep_baseline,LR=1e-4,NUM_EPOCHS=3,MAX_CLIPS=5000,MAX_ULTRACHAT=50000,NUM_LAYERS=1,BLOCK_SIZE=4" \
    -o $SCRATCH/runs/sweep_logs/baseline.log \
    -N dflash_baseline \
    $JOB_SCRIPT)
echo "Exp1 (baseline):     layers=1 bs=4  lr=1e-4  $JID1"

# --- Experiment 2: Higher LR ---
JID2=$(qsub -v "OUTPUT_DIR=$SCRATCH/runs/sweep_lr5e4,LR=5e-4,NUM_EPOCHS=3,MAX_CLIPS=5000,MAX_ULTRACHAT=50000,NUM_LAYERS=1,BLOCK_SIZE=4" \
    -o $SCRATCH/runs/sweep_logs/lr5e4.log \
    -N dflash_lr5e4 \
    $JOB_SCRIPT)
echo "Exp2 (lr=5e-4):      layers=1 bs=4  lr=5e-4  $JID2"

# --- Experiment 3: Lower LR ---
JID3=$(qsub -v "OUTPUT_DIR=$SCRATCH/runs/sweep_lr1e5,LR=1e-5,NUM_EPOCHS=3,MAX_CLIPS=5000,MAX_ULTRACHAT=50000,NUM_LAYERS=1,BLOCK_SIZE=4" \
    -o $SCRATCH/runs/sweep_logs/lr1e5.log \
    -N dflash_lr1e5 \
    $JOB_SCRIPT)
echo "Exp3 (lr=1e-5):      layers=1 bs=4  lr=1e-5  $JID3"

# --- Experiment 4: Larger block (block_size=8) ---
JID4=$(qsub -v "OUTPUT_DIR=$SCRATCH/runs/sweep_bs8,LR=1e-4,NUM_EPOCHS=3,MAX_CLIPS=5000,MAX_ULTRACHAT=50000,NUM_LAYERS=1,BLOCK_SIZE=8" \
    -o $SCRATCH/runs/sweep_logs/bs8.log \
    -N dflash_bs8 \
    $JOB_SCRIPT)
echo "Exp4 (bs=8):         layers=1 bs=8  lr=1e-4  $JID4"

# --- Experiment 5: Larger block (block_size=16) ---
JID5=$(qsub -v "OUTPUT_DIR=$SCRATCH/runs/sweep_bs16,LR=1e-4,NUM_EPOCHS=3,MAX_CLIPS=5000,MAX_ULTRACHAT=50000,NUM_LAYERS=1,BLOCK_SIZE=16" \
    -o $SCRATCH/runs/sweep_logs/bs16.log \
    -N dflash_bs16 \
    $JOB_SCRIPT)
echo "Exp5 (bs=16):        layers=1 bs=16 lr=1e-4  $JID5"

# --- Experiment 6: 2 draft layers ---
JID6=$(qsub -v "OUTPUT_DIR=$SCRATCH/runs/sweep_2layers,LR=1e-4,NUM_EPOCHS=3,MAX_CLIPS=5000,MAX_ULTRACHAT=50000,NUM_LAYERS=2,BLOCK_SIZE=4" \
    -o $SCRATCH/runs/sweep_logs/2layers.log \
    -N dflash_2layers \
    $JOB_SCRIPT)
echo "Exp6 (2 layers):     layers=2 bs=4  lr=1e-4  $JID6"

# --- Experiment 7: 5 draft layers (paper default) ---
JID7=$(qsub -v "OUTPUT_DIR=$SCRATCH/runs/sweep_5layers,LR=1e-4,NUM_EPOCHS=3,MAX_CLIPS=5000,MAX_ULTRACHAT=50000,NUM_LAYERS=5,BLOCK_SIZE=4" \
    -o $SCRATCH/runs/sweep_logs/5layers.log \
    -N dflash_5layers \
    $JOB_SCRIPT)
echo "Exp7 (5 layers):     layers=5 bs=4  lr=1e-4  $JID7"

# --- Experiment 8: 5 layers + block_size=8 (paper-like config) ---
JID8=$(qsub -v "OUTPUT_DIR=$SCRATCH/runs/sweep_5l_bs8,LR=1e-4,NUM_EPOCHS=3,MAX_CLIPS=5000,MAX_ULTRACHAT=50000,NUM_LAYERS=5,BLOCK_SIZE=8" \
    -o $SCRATCH/runs/sweep_logs/5l_bs8.log \
    -N dflash_5l_bs8 \
    $JOB_SCRIPT)
echo "Exp8 (5L+bs8):       layers=5 bs=8  lr=1e-4  $JID8"

echo ""
echo "=== 8 experiments submitted ==="
echo "Monitor: qstat -u z3552416"
echo "Logs: $SCRATCH/runs/sweep_logs/"
echo ""
echo "Sweep grid:"
echo "  LR:          {1e-5, 1e-4, 5e-4}"
echo "  Block size:  {4, 8, 16}"
echo "  Layers:      {1, 2, 5}"

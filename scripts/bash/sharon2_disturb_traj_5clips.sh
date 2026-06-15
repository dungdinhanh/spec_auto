#!/bin/bash
# Run disturb_trajectory_history.py on the 5 source clips used in the CoC modification experiment.
set -e

PYBIN=/home/ubuntu/miniconda3/envs/alpamayo/bin/python
SCRIPT=/home/ubuntu/alpamayo_code/scripts/disturb_trajectory_history.py
NFS=/mnt/resv-harry-6f72s/dungda
TARGET=$NFS/models/Alpamayo-R1-10B
PROC=$NFS/models/Qwen3-VL-2B-Instruct
COC_DIR=$NFS/runs/target_coc_outputs
RAW_DIR=$NFS/data/alpamayo_clips
OUT=$NFS/runs/traj_disturb_viz

export PYTHONPATH=/home/ubuntu/alpamayo_code/src:/home/ubuntu/dflash_code
export VLM_PATH=$NFS/models/Qwen3-VL-8B-Instruct
export PROCESSOR_PATH=$PROC

CLIPS=(
  "0005ec1b-2d1d-45ff-a5c8-129eca592656"
  "000a9fa8-bd18-49d8-b143-6b50c39b7ce7"
  "001564ce-0019-4ec6-bb62-07ed2bd90f2e"
  "001b0192-c2ab-4904-a7e8-3aa49e79ee3c"
  "00f2e502-9fba-43a1-9eb3-4bed06862570"
)

mkdir -p $OUT
LOG=/tmp/traj_disturb_5clips.log
> $LOG

for CLIP in "${CLIPS[@]}"; do
  if [ ! -f "$COC_DIR/$CLIP.pt" ]; then echo "[skip] $CLIP no COC" | tee -a $LOG; continue; fi
  if [ ! -f "$RAW_DIR/$CLIP.pt" ]; then echo "[skip] $CLIP no RAW" | tee -a $LOG; continue; fi
  echo "=== $CLIP ===" | tee -a $LOG
  CUDA_VISIBLE_DEVICES=0 $PYBIN $SCRIPT \
    --target_path $TARGET --processor_path $PROC \
    --target_outputs_dir $COC_DIR --alpamayo_clips_dir $RAW_DIR \
    --clip_uuid $CLIP --out_dir $OUT 2>&1 | tee -a $LOG
done
echo "All done. Outputs in $OUT" | tee -a $LOG

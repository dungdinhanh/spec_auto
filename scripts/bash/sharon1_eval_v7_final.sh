#!/bin/bash
# Eval draft_final.pt for a v7 run on val/test/off-shelf at T=1 (one-shot).
# Usage: sharon1_eval_v7_final.sh <run_dir> <num_draft_layers> <gpu_base>
#   <gpu_base> = first of 3 consecutive GPUs (e.g. 0 → uses 0,1,2).
set -e
RUN_DIR=$1
NLAYERS=$2
G=$3

LOCAL=/home/ubuntu/local_data
SCRIPTS=/home/ubuntu/katana_transfer/code/claude_mod
SPLITS=/home/ubuntu/katana_transfer/splits
OFFSHELF=/home/ubuntu/katana_transfer/offshelf/alpamayo_clips_offshelf
OUT=$RUN_DIR/_eval_1d
mkdir -p $OUT

TARGET=$LOCAL/models/Alpamayo-R1-10B
COC=$LOCAL/runs/target_coc_outputs
DRAFT=$RUN_DIR/draft_final.pt

PYBIN=/home/ubuntu/alpamayo_env/bin/python
export PYTHONPATH=/home/ubuntu/dflash_code:/home/ubuntu/katana_transfer/code/src
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

if [ ! -f "$DRAFT" ]; then
  echo "ERROR: draft_final.pt missing in $RUN_DIR"; exit 1
fi

CUDA_VISIBLE_DEVICES=$G nohup $PYBIN $SCRIPTS/e2e_spec_test.py \
  --target_path $TARGET --draft_path $DRAFT \
  --clips_dir $COC --uuids_file $SPLITS/val_uuids_v3.json \
  --num_draft_layers $NLAYERS --block_size 16 --num_target_features 5 \
  --output_json $OUT/val.json > $OUT/val.log 2>&1 &
P0=$!

CUDA_VISIBLE_DEVICES=$((G+1)) nohup $PYBIN $SCRIPTS/e2e_spec_test.py \
  --target_path $TARGET --draft_path $DRAFT \
  --clips_dir $COC --uuids_file $SPLITS/test_uuids_v3.json \
  --num_draft_layers $NLAYERS --block_size 16 --num_target_features 5 \
  --output_json $OUT/test.json > $OUT/test.log 2>&1 &
P1=$!

CUDA_VISIBLE_DEVICES=$((G+2)) nohup $PYBIN $SCRIPTS/e2e_spec_test.py \
  --target_path $TARGET --draft_path $DRAFT \
  --clips_dir $OFFSHELF --uuids_file $OFFSHELF/test_offshelf_uuids.json \
  --num_draft_layers $NLAYERS --block_size 16 --num_target_features 5 \
  --output_json $OUT/offshelf.json > $OUT/offshelf.log 2>&1 &
P2=$!

echo "EVAL_LAUNCHED $(basename $RUN_DIR) pids=$P0,$P1,$P2"
wait
echo "EVAL_DONE $(basename $RUN_DIR) $(date)"
for f in $OUT/*.log; do
  tag=$(basename $f .log)
  echo "$tag: $(grep 'Avg tokens' $f)"
done

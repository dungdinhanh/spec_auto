#!/bin/bash
# Eval warm SFT v6 L=4 ep50 + random_mask draft_final on val/test/off-shelf.
# 1D FlatRoPE, num_target_features=5. 3 splits in parallel on GPUs 0-2.
set -e

LOCAL=/home/ubuntu/local_data
SCRIPTS=/home/ubuntu/katana_transfer/code/claude_mod
SPLITS=/home/ubuntu/katana_transfer/splits
OFFSHELF=/home/ubuntu/katana_transfer/offshelf/alpamayo_clips_offshelf
RUN=$LOCAL/runs/dflash_L4_lr1e-4_ep50_bs16_warm_v6_randomMask_sharon1
OUT=$RUN/_eval_1d
mkdir -p $OUT

TARGET=$LOCAL/models/Alpamayo-R1-10B
COC=$LOCAL/runs/target_coc_outputs
DRAFT=$RUN/draft_final.pt

PYBIN=/home/ubuntu/alpamayo_env/bin/python
export PYTHONPATH=/home/ubuntu/dflash_code:/home/ubuntu/katana_transfer/code/src
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

if [ ! -f "$DRAFT" ]; then
  echo "ERROR: draft_final.pt missing in $RUN"
  exit 1
fi

CUDA_VISIBLE_DEVICES=0 nohup $PYBIN $SCRIPTS/e2e_spec_test.py \
  --target_path $TARGET --draft_path $DRAFT \
  --clips_dir $COC --uuids_file $SPLITS/val_uuids_v3.json \
  --num_draft_layers 4 --block_size 16 --num_target_features 5 \
  --output_json $OUT/val.json > $OUT/val.log 2>&1 &
P0=$!

CUDA_VISIBLE_DEVICES=1 nohup $PYBIN $SCRIPTS/e2e_spec_test.py \
  --target_path $TARGET --draft_path $DRAFT \
  --clips_dir $COC --uuids_file $SPLITS/test_uuids_v3.json \
  --num_draft_layers 4 --block_size 16 --num_target_features 5 \
  --output_json $OUT/test.json > $OUT/test.log 2>&1 &
P1=$!

CUDA_VISIBLE_DEVICES=2 nohup $PYBIN $SCRIPTS/e2e_spec_test.py \
  --target_path $TARGET --draft_path $DRAFT \
  --clips_dir $OFFSHELF --uuids_file $OFFSHELF/test_offshelf_uuids.json \
  --num_draft_layers 4 --block_size 16 --num_target_features 5 \
  --output_json $OUT/offshelf.json > $OUT/offshelf.log 2>&1 &
P2=$!

echo "EVAL_V6_L4_EP50_RANDOMMASK_LAUNCHED pids=$P0,$P1,$P2"
wait
echo "EVAL_V6_L4_EP50_RANDOMMASK_DONE $(date)"
echo "--- summary ---"
for f in $OUT/*.log; do
  tag=$(basename $f .log)
  echo "$tag: $(grep 'Avg tokens' $f)"
done

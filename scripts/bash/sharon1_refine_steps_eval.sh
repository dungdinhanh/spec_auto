#!/bin/bash
# Iterative-refinement eval grid: T in {1,2,3,5} for best L=2 / L=4 RM ckpts on val/test/off-shelf.
# Best ckpts (by off-shelf L from v6 RM sweep): L=4 ep36, L=2 ep19.
# 2 models × 4 T values × 3 splits = 24 evals, round-robin across 8 GPUs.
set -e

LOCAL=/home/ubuntu/local_data
SCRIPTS=/home/ubuntu/katana_transfer/code/claude_mod
SPLITS=/home/ubuntu/katana_transfer/splits
OFFSHELF=/home/ubuntu/katana_transfer/offshelf/alpamayo_clips_offshelf
TARGET=$LOCAL/models/Alpamayo-R1-10B
COC=$LOCAL/runs/target_coc_outputs

PYBIN=/home/ubuntu/alpamayo_env/bin/python
export PYTHONPATH=/home/ubuntu/dflash_code:/home/ubuntu/katana_transfer/code/src
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

L4_CK=$LOCAL/runs/dflash_L4_lr1e-4_ep50_bs16_warm_v6_randomMask_sharon1/draft_epoch_36.pt
L2_CK=$LOCAL/runs/dflash_L2_lr1e-4_ep50_bs16_warm_v6_randomMask_sharon1/draft_epoch_19.pt

OUT=$LOCAL/runs/_refine_eval
mkdir -p $OUT

JOBS=$(mktemp)
for T in 1 2 3 5; do
  for cfg in "L4|4|$L4_CK" "L2|2|$L2_CK"; do
    name=${cfg%%|*}; rest=${cfg#*|}; nlayers=${rest%%|*}; ck=${rest##*|}
    echo "$name|$nlayers|$ck|$COC|$SPLITS/val_uuids_v3.json|$T|$OUT/${name}_T${T}_val" >> $JOBS
    echo "$name|$nlayers|$ck|$COC|$SPLITS/test_uuids_v3.json|$T|$OUT/${name}_T${T}_test" >> $JOBS
    echo "$name|$nlayers|$ck|$OFFSHELF|$OFFSHELF/test_offshelf_uuids.json|$T|$OUT/${name}_T${T}_offshelf" >> $JOBS
  done
done
N=$(wc -l < $JOBS); echo "Total jobs: $N"

run_gpu() {
  local gpu=$1
  while IFS='|' read -r name nlayers ck cdir uuids T outpref; do
    [ -z "$ck" ] && continue
    if [ -f "${outpref}.json" ]; then
      echo "[gpu$gpu] SKIP $(basename $outpref)"
      continue
    fi
    echo "[gpu$gpu] RUN $(basename $outpref)"
    CUDA_VISIBLE_DEVICES=$gpu $PYBIN $SCRIPTS/e2e_spec_test.py \
      --target_path $TARGET --draft_path $ck \
      --clips_dir $cdir --uuids_file $uuids \
      --num_draft_layers $nlayers --block_size 16 --num_target_features 5 \
      --refinement_steps $T \
      --output_json ${outpref}.json > ${outpref}.log 2>&1
  done
}
export -f run_gpu
export PYBIN SCRIPTS TARGET

for g in 0 1 2 3 4 5 6 7; do
  awk -v g=$g -v n=8 'NR % n == g' $JOBS | run_gpu $g > /tmp/refine_gpu${g}.out 2>&1 &
done
wait
rm -f $JOBS
echo "REFINE_DONE $(date)"

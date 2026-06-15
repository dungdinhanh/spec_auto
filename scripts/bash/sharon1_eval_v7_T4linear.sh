#!/bin/bash
# Eval v7 L=2 best-val ckpts with T=4 linear refinement (matches training mask
# states {15,11,8,4}). varB also locks pos 1 through steps 1..3, mirroring its
# --always_mask_pos1 training. Comparison set:
#   varA ep25  ×  {T=1, T=4 linear}            ×  {val, test, offshelf}
#   varB ep31  ×  {T=1, T=4 linear + lock_pos1} ×  {val, test, offshelf}
# = 12 evals, round-robin across 8 GPUs.
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

A_RUN=$LOCAL/runs/dflash_L2_lr1e-4_ep50_bs16_warm_v7_varA_sharon1
B_RUN=$LOCAL/runs/dflash_L2_lr1e-4_ep50_bs16_warm_v7_varB_sharon1
A_CK=$A_RUN/draft_epoch_25.pt
B_CK=$B_RUN/draft_epoch_31.pt
OUT=$LOCAL/runs/_v7_T4_eval
mkdir -p $OUT

JOBS=$(mktemp)
# var|ckpt|T|lock|cdir|uuids|outpref
for SPLIT in "val|$COC|$SPLITS/val_uuids_v3.json" \
             "test|$COC|$SPLITS/test_uuids_v3.json" \
             "offshelf|$OFFSHELF|$OFFSHELF/test_offshelf_uuids.json"; do
  name=${SPLIT%%|*}; rest=${SPLIT#*|}; cdir=${rest%%|*}; uuids=${rest##*|}
  echo "A|$A_CK|1|0|$cdir|$uuids|$OUT/varA_T1_$name" >> $JOBS
  echo "A|$A_CK|4|0|$cdir|$uuids|$OUT/varA_T4lin_$name" >> $JOBS
  echo "B|$B_CK|1|0|$cdir|$uuids|$OUT/varB_T1_$name" >> $JOBS
  echo "B|$B_CK|4|1|$cdir|$uuids|$OUT/varB_T4lin_lock_$name" >> $JOBS
done

run_gpu() {
  local g=$1
  while IFS='|' read -r var ck T lock cdir uuids outpref; do
    [ -z "$ck" ] && continue
    if [ -f "${outpref}.json" ]; then echo "[gpu$g] SKIP $(basename $outpref)"; continue; fi
    EXTRA="--refinement_steps $T"
    [ "$T" -gt 1 ] && EXTRA="$EXTRA --refinement_schedule linear"
    [ "$lock" = "1" ] && EXTRA="$EXTRA --lock_pos1"
    echo "[gpu$g] RUN $(basename $outpref) $EXTRA"
    CUDA_VISIBLE_DEVICES=$g $PYBIN $SCRIPTS/e2e_spec_test.py \
      --target_path $TARGET --draft_path $ck \
      --clips_dir $cdir --uuids_file $uuids \
      --num_draft_layers 2 --block_size 16 --num_target_features 5 \
      $EXTRA \
      --output_json ${outpref}.json > ${outpref}.log 2>&1
  done
}
export -f run_gpu
export PYBIN SCRIPTS TARGET

for g in 0 1 2 3 4 5 6 7; do
  awk -v g=$g -v n=8 'NR % n == g' $JOBS | run_gpu $g > /tmp/v7_T4_g${g}.out 2>&1 &
done
wait
rm -f $JOBS
echo "DONE $(date)"

echo "--- summary ---"
for f in $OUT/*.json; do
  L=$(python3 -c "import json; print(round(json.load(open('$f'))['avg_iter_tokens'], 4))" 2>/dev/null)
  echo "$(basename $f .json): L=$L"
done

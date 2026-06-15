#!/bin/bash
# Val-only sweep over surviving v7 L=2 ckpts (ep25-30 varA, ep26-31 varB)
# to check if the model converged or is still climbing.
# 12 jobs, distributed round-robin across 8 GPUs.
set -e

LOCAL=/home/ubuntu/local_data
SCRIPTS=/home/ubuntu/katana_transfer/code/claude_mod
SPLITS=/home/ubuntu/katana_transfer/splits
TARGET=$LOCAL/models/Alpamayo-R1-10B
COC=$LOCAL/runs/target_coc_outputs

PYBIN=/home/ubuntu/alpamayo_env/bin/python
export PYTHONPATH=/home/ubuntu/dflash_code:/home/ubuntu/katana_transfer/code/src
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

A_RUN=$LOCAL/runs/dflash_L2_lr1e-4_ep50_bs16_warm_v7_varA_sharon1
B_RUN=$LOCAL/runs/dflash_L2_lr1e-4_ep50_bs16_warm_v7_varB_sharon1
mkdir -p $A_RUN/_ep_sweep $B_RUN/_ep_sweep

JOBS=$(mktemp)
for ep in 25 26 27 28 29 30; do
  echo "A|$ep|$A_RUN/draft_epoch_${ep}.pt|$A_RUN/_ep_sweep/val_ep${ep}" >> $JOBS
done
for ep in 26 27 28 29 30 31; do
  echo "B|$ep|$B_RUN/draft_epoch_${ep}.pt|$B_RUN/_ep_sweep/val_ep${ep}" >> $JOBS
done

run_gpu() {
  local g=$1
  while IFS='|' read -r var ep ck out; do
    [ -z "$ck" ] && continue
    if [ -f "${out}.json" ]; then echo "[gpu$g] SKIP $var ep$ep"; continue; fi
    echo "[gpu$g] RUN $var ep$ep"
    CUDA_VISIBLE_DEVICES=$g $PYBIN $SCRIPTS/e2e_spec_test.py \
      --target_path $TARGET --draft_path $ck \
      --clips_dir $COC --uuids_file $SPLITS/val_uuids_v3.json \
      --num_draft_layers 2 --block_size 16 --num_target_features 5 \
      --output_json ${out}.json > ${out}.log 2>&1
  done
}
export -f run_gpu
export PYBIN SCRIPTS TARGET COC SPLITS

for g in 0 1 2 3 4 5 6 7; do
  awk -v g=$g -v n=8 'NR % n == g' $JOBS | run_gpu $g > /tmp/v7_sweep_g${g}.out 2>&1 &
done
wait
rm -f $JOBS
echo "DONE $(date)"

echo "--- summary ---"
for var in A B; do
  if [ "$var" = "A" ]; then RUN=$A_RUN; EPS="25 26 27 28 29 30"; else RUN=$B_RUN; EPS="26 27 28 29 30 31"; fi
  echo "=== var${var} ==="
  for ep in $EPS; do
    L=$(python3 -c "import json; print(round(json.load(open('$RUN/_ep_sweep/val_ep${ep}.json'))['avg_iter_tokens'], 4))" 2>/dev/null)
    echo "  ep${ep}: L=$L"
  done
done

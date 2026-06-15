#!/bin/bash
# Sweep every epoch ckpt of v6 ep50 (L=2 and L=4) on val/test/off-shelf.
# 50 ckpts × 2 runs × 3 splits = 300 evals, distributed across 8 GPUs.
# Each GPU runs its job queue sequentially; e2e_spec_test.py invoked per job.
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

L4_RUN=$LOCAL/runs/dflash_L4_lr1e-4_ep50_bs16_warm_v6_sharon1
L2_RUN=$LOCAL/runs/dflash_L2_lr1e-4_ep50_bs16_warm_v6_sharon1
mkdir -p $L4_RUN/_eval_sweep $L2_RUN/_eval_sweep

# Build job list as lines: ckpt|nlayers|clips_dir|uuids_file|out_prefix
JOBS_FILE=$(mktemp)
for ep in $(seq 1 50); do
  # L=4
  ck=$L4_RUN/draft_epoch_$ep.pt
  if [ -f "$ck" ]; then
    echo "$ck|4|$COC|$SPLITS/val_uuids_v3.json|$L4_RUN/_eval_sweep/epoch_${ep}_val" >> $JOBS_FILE
    echo "$ck|4|$COC|$SPLITS/test_uuids_v3.json|$L4_RUN/_eval_sweep/epoch_${ep}_test" >> $JOBS_FILE
    echo "$ck|4|$OFFSHELF|$OFFSHELF/test_offshelf_uuids.json|$L4_RUN/_eval_sweep/epoch_${ep}_offshelf" >> $JOBS_FILE
  fi
  # L=2
  ck=$L2_RUN/draft_epoch_$ep.pt
  if [ -f "$ck" ]; then
    echo "$ck|2|$COC|$SPLITS/val_uuids_v3.json|$L2_RUN/_eval_sweep/epoch_${ep}_val" >> $JOBS_FILE
    echo "$ck|2|$COC|$SPLITS/test_uuids_v3.json|$L2_RUN/_eval_sweep/epoch_${ep}_test" >> $JOBS_FILE
    echo "$ck|2|$OFFSHELF|$OFFSHELF/test_offshelf_uuids.json|$L2_RUN/_eval_sweep/epoch_${ep}_offshelf" >> $JOBS_FILE
  fi
done
N_JOBS=$(wc -l < $JOBS_FILE)
echo "Total jobs: $N_JOBS"
N_GPUS=8

# Per-GPU dispatcher: reads its slice of jobs from stdin, runs each in turn.
run_gpu() {
  local gpu=$1
  while IFS='|' read -r ck nlayers cdir uuids outpref; do
    [ -z "$ck" ] && continue
    if [ -f "${outpref}.json" ]; then
      echo "[gpu$gpu] SKIP $(basename $outpref) (already done)"
      continue
    fi
    echo "[gpu$gpu] RUN $(basename $outpref) ckpt=$(basename $ck) layers=$nlayers"
    CUDA_VISIBLE_DEVICES=$gpu $PYBIN $SCRIPTS/e2e_spec_test.py \
      --target_path $TARGET --draft_path $ck \
      --clips_dir $cdir --uuids_file $uuids \
      --num_draft_layers $nlayers --block_size 16 --num_target_features 5 \
      --output_json ${outpref}.json > ${outpref}.log 2>&1
  done
}
export -f run_gpu
export PYBIN SCRIPTS TARGET

# Stripe jobs across 8 GPUs (round-robin so each GPU gets a mix of L=2 and L=4)
for g in $(seq 0 $((N_GPUS-1))); do
  awk -v g=$g -v n=$N_GPUS 'NR % n == g' $JOBS_FILE | run_gpu $g > /tmp/sweep_gpu${g}.out 2>&1 &
  echo "Launched GPU $g (jobs: $(awk -v g=$g -v n=$N_GPUS 'NR % n == g' $JOBS_FILE | wc -l))"
done
wait
rm -f $JOBS_FILE
echo "SWEEP_DONE $(date)"

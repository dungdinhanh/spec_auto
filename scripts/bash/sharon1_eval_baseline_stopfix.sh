#!/bin/bash
# Re-eval v6-RM L=2 ep19 and L=4 ep36 BASELINE (T=1) on val/test/off-shelf
# after the stop-check off-by-one fix in vlm_spec_generate.
# 6 evals, 1 GPU each, parallel.
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

L2_CK=$LOCAL/runs/dflash_L2_lr1e-4_ep50_bs16_warm_v6_randomMask_sharon1/draft_epoch_19.pt
L4_CK=$LOCAL/runs/dflash_L4_lr1e-4_ep50_bs16_warm_v6_randomMask_sharon1/draft_epoch_36.pt
OUT=$LOCAL/runs/_baseline_stopfix_eval
mkdir -p $OUT

run_eval() {
  local gpu=$1 ck=$2 nlayers=$3 cdir=$4 uuids=$5 out=$6
  CUDA_VISIBLE_DEVICES=$gpu $PYBIN $SCRIPTS/e2e_spec_test.py \
    --target_path $TARGET --draft_path $ck \
    --clips_dir $cdir --uuids_file $uuids \
    --num_draft_layers $nlayers --block_size 16 --num_target_features 5 \
    --output_json ${out}.json > ${out}.log 2>&1
}

run_eval 0 $L2_CK 2 $COC $SPLITS/val_uuids_v3.json $OUT/L2_val &
run_eval 1 $L2_CK 2 $COC $SPLITS/test_uuids_v3.json $OUT/L2_test &
run_eval 2 $L2_CK 2 $OFFSHELF $OFFSHELF/test_offshelf_uuids.json $OUT/L2_offshelf &
run_eval 4 $L4_CK 4 $COC $SPLITS/val_uuids_v3.json $OUT/L4_val &
run_eval 5 $L4_CK 4 $COC $SPLITS/test_uuids_v3.json $OUT/L4_test &
run_eval 6 $L4_CK 4 $OFFSHELF $OFFSHELF/test_offshelf_uuids.json $OUT/L4_offshelf &
wait

echo "DONE $(date)"
echo "--- summary ---"
for f in $OUT/*.json; do
  L=$(python3 -c "import json; d=json.load(open('$f')); print(round(d['avg_iter_tokens'],4), round(d['speedup'],3))" 2>/dev/null)
  echo "$(basename $f .json): L=$L (avg_tok speedup)"
done

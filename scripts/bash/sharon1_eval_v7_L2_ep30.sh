#!/bin/bash
# Eval v7 L=2 last-good ckpts (varA ep30, varB ep31) on val/test/off-shelf at T=1.
# Both ckpts are the highest healthy epoch before the disk-full crash.
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
A_CK=$A_RUN/draft_epoch_30.pt
B_CK=$B_RUN/draft_epoch_31.pt
mkdir -p $A_RUN/_eval_1d $B_RUN/_eval_1d

run_eval() {
  local gpu=$1 ck=$2 nlayers=$3 cdir=$4 uuids=$5 out=$6
  CUDA_VISIBLE_DEVICES=$gpu $PYBIN $SCRIPTS/e2e_spec_test.py \
    --target_path $TARGET --draft_path $ck \
    --clips_dir $cdir --uuids_file $uuids \
    --num_draft_layers $nlayers --block_size 16 --num_target_features 5 \
    --output_json ${out}.json > ${out}.log 2>&1
}

# varA: GPU 0,1,2
run_eval 0 $A_CK 2 $COC $SPLITS/val_uuids_v3.json $A_RUN/_eval_1d/val &
run_eval 1 $A_CK 2 $COC $SPLITS/test_uuids_v3.json $A_RUN/_eval_1d/test &
run_eval 2 $A_CK 2 $OFFSHELF $OFFSHELF/test_offshelf_uuids.json $A_RUN/_eval_1d/offshelf &
# varB: GPU 4,5,6
run_eval 4 $B_CK 2 $COC $SPLITS/val_uuids_v3.json $B_RUN/_eval_1d/val &
run_eval 5 $B_CK 2 $COC $SPLITS/test_uuids_v3.json $B_RUN/_eval_1d/test &
run_eval 6 $B_CK 2 $OFFSHELF $OFFSHELF/test_offshelf_uuids.json $B_RUN/_eval_1d/offshelf &
wait
echo "DONE $(date)"
for r in $A_RUN $B_RUN; do
  echo "=== $(basename $r) ==="
  for s in val test offshelf; do
    L=$(python3 -c "import json; print(json.load(open('$r/_eval_1d/$s.json'))['avg_iter_tokens'])" 2>/dev/null)
    echo "  $s: L=$L"
  done
done

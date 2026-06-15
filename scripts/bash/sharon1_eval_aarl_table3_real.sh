#!/bin/bash
# Re-eval AARL ablations (paper Table 3) using e2e_spec_test.py (real L).
# 3 models × 3 splits = 9 evals across GPUs 0-7 (one GPU runs 2 sequentially).
set -e

LOCAL=/home/ubuntu/local_data
SCRIPTS=/home/ubuntu/katana_transfer/code/claude_mod
SPLITS=/home/ubuntu/katana_transfer/splits
OFFSHELF=/home/ubuntu/katana_transfer/offshelf/alpamayo_clips_offshelf
OUT=$LOCAL/runs/aarl_table3_real_eval
mkdir -p "$OUT"

TARGET=$LOCAL/models/Alpamayo-R1-10B
COC=$LOCAL/runs/target_coc_outputs
VAL=$SPLITS/val_uuids_v3.json
TEST=$SPLITS/test_uuids_v3.json
OUUIDS=$OFFSHELF/test_offshelf_uuids.json

PYBIN=/home/ubuntu/alpamayo_env/bin/python
export PYTHONPATH=/home/ubuntu/dflash_code:/home/ubuntu/katana_transfer/code/src
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

POLICY=$LOCAL/runs/rl_v3_optionA_K5/draft_final.pt          # policy anchor, K=5
K10=$LOCAL/runs/rl_v3_optA_ref_K10/draft_final.pt           # static-ref, K=10
K15=$LOCAL/runs/rl_v3_optA_ref_K15/draft_final.pt           # static-ref, K=15

JOBS=(
  "0|$POLICY|$VAL|$COC|policy_val"
  "1|$POLICY|$TEST|$COC|policy_test"
  "2|$POLICY|$OUUIDS|$OFFSHELF|policy_offshelf"
  "3|$K10|$VAL|$COC|K10_val"
  "4|$K10|$TEST|$COC|K10_test"
  "5|$K10|$OUUIDS|$OFFSHELF|K10_offshelf"
  "6|$K15|$VAL|$COC|K15_val"
  "7|$K15|$TEST|$COC|K15_test"
  "0|$K15|$OUUIDS|$OFFSHELF|K15_offshelf"     # GPU 0 takes a 2nd job
)

for j in "${JOBS[@]}"; do
  IFS='|' read -r gpu draft uuids clips tag <<< "$j"
  log=$OUT/${tag}.log
  json=$OUT/${tag}.json
  CUDA_VISIBLE_DEVICES=$gpu nohup $PYBIN $SCRIPTS/e2e_spec_test.py \
    --target_path $TARGET --draft_path $draft \
    --clips_dir $clips --uuids_file $uuids \
    --num_draft_layers 4 --block_size 16 \
    --output_json $json > $log 2>&1 &
  echo "GPU $gpu launched (pid=$!) tag=$tag"
  sleep 0.5   # space launches to avoid simultaneous model-load thrashing
done

sleep 2
echo "running: $(ps -ef | grep e2e_spec_test | grep -v grep | wc -l)"
wait
echo "AARL_TABLE3_REAL_DONE $(date)"

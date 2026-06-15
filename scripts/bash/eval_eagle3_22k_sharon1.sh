#!/bin/bash
# Eval EAGLE-3 22k 1D + 3D on sharon1 — answer the "L=5.x vs paper 7" question.
# 6 evals parallel on GPUs 0-5, gamma=7, ep20 (final).
set -e
source /home/ubuntu/alpamayo_env/bin/activate
export PYTHONPATH=/home/ubuntu/katana_transfer/code/src
export VLM_PATH=/home/ubuntu/local_data/models/Qwen3-VL-8B-Instruct
export PROCESSOR_PATH=/home/ubuntu/local_data/models/Qwen3-VL-2B-Instruct
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

DATA=/home/ubuntu/local_data
T=$DATA/models/Alpamayo-R1-10B
C=$DATA/runs/target_coc_outputs_combined
S=/home/ubuntu/katana_transfer/splits
CODE=/home/ubuntu/katana_transfer/code
EP=20

OUT=$DATA/runs/eval_eagle3_22k_sharon1
mkdir -p $OUT

run() {
    local gpu=$1 variant=$2 split=$3 uuids=$4
    local draft="$DATA/runs/eagle3_22k_${variant}_ep20_katana/draft_epoch_${EP}.pt"
    local script="claude_mod/e2e_eagle3_spec_test.py"
    [ "$variant" = "3d" ] && script="claude_mod/e2e_eagle3_spec_test_3d.py"
    local out_json="$OUT/${variant}_${split}_ep${EP}.json"
    local log="$OUT/${variant}_${split}_ep${EP}.log"
    echo "[GPU $gpu] $variant $split ep$EP -> $log"
    CUDA_VISIBLE_DEVICES=$gpu nohup python $CODE/$script \
        --target_path $T --draft_path $draft --clips_dir $C \
        --uuids_file $uuids --gamma 7 --max_new_tokens 128 \
        --output_json $out_json > $log 2>&1 &
}

cd $CODE
run 0 1d train  $S/train100_uuids_v3.json
run 1 1d val    $S/val_uuids_v3.json
run 2 1d test   $S/test_uuids_v3.json
run 3 3d train  $S/train100_uuids_v3.json
run 4 3d val    $S/val_uuids_v3.json
run 5 3d test   $S/test_uuids_v3.json
wait
echo "DONE_EAGLE3_22K_EVAL $(date)"

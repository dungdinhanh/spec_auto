#!/bin/bash
# Eval J5_final, J9_step3500, J9_final on sharon1's 14k pool for direct
# apples-to-apples comparison with sharon1 v4/v5 AARL numbers.
set -e
source /home/ubuntu/alpamayo_env/bin/activate
export PYTHONPATH=/home/ubuntu/katana_transfer/code/src
export VLM_PATH=/home/ubuntu/local_data/models/Qwen3-VL-8B-Instruct
export PROCESSOR_PATH=/home/ubuntu/local_data/models/Qwen3-VL-2B-Instruct
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

DATA=/home/ubuntu/local_data
T=$DATA/models/Alpamayo-R1-10B
COC=$DATA/runs/target_coc_outputs_combined
S=/home/ubuntu/katana_transfer/splits
CODE=/home/ubuntu/katana_transfer/code
CKPT_DIR=$DATA/runs/_katana_aarl_ckpts

OUT=$DATA/runs/_katana_aarl_eval_on_sharon1
mkdir -p $OUT

ev(){
    local gpu=$1 tag=$2 split=$3 uuids=$4 draft=$5
    local oj=$OUT/${tag}_${split}.json
    CUDA_VISIBLE_DEVICES=$gpu nohup python $CODE/claude_mod/e2e_spec_test.py \
        --target_path $T --draft_path $draft --clips_dir $COC --uuids_file $uuids \
        --num_draft_layers 2 --block_size 16 --num_target_features 5 \
        --output_json $oj > ${oj%.json}.log 2>&1 &
}

ev 0 J5_final     val  $S/val_uuids_v3.json  $CKPT_DIR/J5_final.pt
ev 1 J5_final     test $S/test_uuids_v3.json $CKPT_DIR/J5_final.pt
ev 2 J9_step3500  val  $S/val_uuids_v3.json  $CKPT_DIR/J9_step3500.pt
ev 3 J9_step3500  test $S/test_uuids_v3.json $CKPT_DIR/J9_step3500.pt
ev 4 J9_final     val  $S/val_uuids_v3.json  $CKPT_DIR/J9_final.pt
ev 5 J9_final     test $S/test_uuids_v3.json $CKPT_DIR/J9_final.pt
wait
echo "DONE_KATANA_AARL_EVAL_SHARON1 $(date)"
for f in $OUT/*.json; do
    L=$(python -c "import json;print(round(json.load(open('$f'))['avg_iter_tokens'],4))" 2>/dev/null)
    sp=$(python -c "import json;print(round(json.load(open('$f'))['speedup'],4))" 2>/dev/null)
    echo "$(basename $f .json): L=$L speedup=$sp"
done

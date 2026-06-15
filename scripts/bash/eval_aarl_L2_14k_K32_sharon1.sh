#!/bin/bash
# Eval AARL K=32 ckpts vs init on val/test — answer: does GRPO lift held-out L?
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

INIT=$DATA/runs/dflash_L2_probe_B_combined_ep20_ceonly_sharon1/draft_epoch_15.pt
S15=$DATA/runs/aarl_L2_14k_K32_ep2_sharon1/draft_step_1500.pt
FIN=$DATA/runs/aarl_L2_14k_K32_ep2_sharon1/draft_final.pt

OUT=$DATA/runs/aarl_L2_14k_K32_ep2_sharon1/_eval
mkdir -p $OUT

ev(){
    local gpu=$1 tag=$2 split=$3 uuids=$4 draft=$5
    local oj=$OUT/${tag}_${split}.json
    CUDA_VISIBLE_DEVICES=$gpu nohup python $CODE/claude_mod/e2e_spec_test.py \
        --target_path $T --draft_path $draft --clips_dir $COC --uuids_file $uuids \
        --num_draft_layers 2 --block_size 16 --num_target_features 5 \
        --output_json $oj > ${oj%.json}.log 2>&1 &
}

ev 0 init      val  $S/val_uuids_v3.json  $INIT
ev 1 init      test $S/test_uuids_v3.json $INIT
ev 2 step1500  val  $S/val_uuids_v3.json  $S15
ev 3 step1500  test $S/test_uuids_v3.json $S15
ev 4 final     val  $S/val_uuids_v3.json  $FIN
ev 5 final     test $S/test_uuids_v3.json $FIN
wait
echo "DONE_AARL_EVAL $(date)"
for f in $OUT/*.json; do
    L=$(python -c "import json;print(round(json.load(open('$f'))['avg_iter_tokens'],4))")
    sp=$(python -c "import json;print(round(json.load(open('$f'))['speedup'],4))")
    echo "$(basename $f .json): L=$L speedup=$sp"
done

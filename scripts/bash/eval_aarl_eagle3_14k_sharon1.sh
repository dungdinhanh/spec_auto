#!/bin/bash
# Eval EAGLE-3 14k AARL ckpts (init + 4 step ckpts + final) on val + test.
# 10 evals on 8 GPUs (2 rounds).
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

INIT=$DATA/runs/eagle3_22k_1d_ep20_katana/draft_final.pt
RUN=$DATA/runs/aarl_eagle3_v2_14k_sharon1_N5_max20
OUT=$RUN/_eval
mkdir -p $OUT

ev(){
    local gpu=$1 tag=$2 split=$3 uuids=$4 draft=$5
    local oj=$OUT/${tag}_${split}.json
    CUDA_VISIBLE_DEVICES=$gpu nohup python $CODE/claude_mod/e2e_eagle3_spec_test.py \
        --target_path $T --draft_path $draft --clips_dir $COC --uuids_file $uuids \
        --gamma 7 --max_new_tokens 128 \
        --output_json $oj > ${oj%.json}.log 2>&1 &
}

# Round 1: init + step ckpts on val (5 evals on 5 GPUs); plus init + 2 ckpts test (3 evals on GPUs 5-7)
ev 0 init      val  $S/val_uuids_v3.json   $INIT
ev 1 step500   val  $S/val_uuids_v3.json   $RUN/draft_step_500.pt
ev 2 step1000  val  $S/val_uuids_v3.json   $RUN/draft_step_1000.pt
ev 3 step1500  val  $S/val_uuids_v3.json   $RUN/draft_step_1500.pt
ev 4 final     val  $S/val_uuids_v3.json   $RUN/draft_final.pt
ev 5 init      test $S/test_uuids_v3.json  $INIT
ev 6 step500   test $S/test_uuids_v3.json  $RUN/draft_step_500.pt
ev 7 step1000  test $S/test_uuids_v3.json  $RUN/draft_step_1000.pt
wait
echo "ROUND1_DONE $(date)"

# Round 2: remaining 2 test evals
ev 0 step1500  test $S/test_uuids_v3.json  $RUN/draft_step_1500.pt
ev 1 final     test $S/test_uuids_v3.json  $RUN/draft_final.pt
wait
echo "DONE_AARL_EAGLE3_14k_EVAL $(date)"
for f in $OUT/*.json; do
    L=$(python -c "import json;print(round(json.load(open('$f'))['avg_iter_tokens'],4))" 2>/dev/null)
    sp=$(python -c "import json;print(round(json.load(open('$f'))['speedup'],4))" 2>/dev/null)
    echo "$(basename $f .json): L=$L speedup=$sp"
done

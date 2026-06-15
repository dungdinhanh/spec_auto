#!/bin/bash
# Eval warm SFT v4 (M-RoPE 3D fix) — sweep key ckpts + final.
# Compare against original warm SFT baseline (val 4.524 / test 4.208).
set -e

LOCAL=/home/ubuntu/local_data
SCRIPTS=/home/ubuntu/katana_transfer/code/scripts
SPLITS=/home/ubuntu/katana_transfer/splits
RUN=dflash_L4_lr1e-4_ep15_bs16_warm_v4_sharon1
OUTDIR=$LOCAL/runs/${RUN}_eval
mkdir -p "$OUTDIR"

TARGET=$LOCAL/models/Alpamayo-R1-10B
COC=$LOCAL/runs/target_coc_outputs
VAL=$SPLITS/val_uuids_v3.json
TEST=$SPLITS/test_uuids_v3.json
RUN_DIR=$LOCAL/runs/$RUN

# Pick representative ckpts: every other epoch + step interleaved + final.
CK_GPU0=( $RUN_DIR/draft_final.pt $RUN_DIR/draft_epoch_15.pt $RUN_DIR/draft_epoch_10.pt )
CK_GPU1=( $RUN_DIR/draft_epoch_5.pt $RUN_DIR/draft_epoch_1.pt )
CK_GPU2=( $RUN_DIR/draft_step_500.pt $RUN_DIR/draft_step_2000.pt )
CK_GPU3=( $RUN_DIR/draft_step_4000.pt $RUN_DIR/draft_step_6000.pt )

PYBIN=/home/ubuntu/alpamayo_env/bin/python
export PYTHONPATH=/home/ubuntu/dflash_code:/home/ubuntu/katana_transfer/code/src

run_chunk() {
  local gpu=$1; shift
  local drafts=("$@")
  local log=$OUTDIR/eval_gpu${gpu}.log
  local csv=$OUTDIR/eval_gpu${gpu}.csv
  CUDA_VISIBLE_DEVICES=$gpu nohup $PYBIN $SCRIPTS/eval_ckpt_sweep_vt.py \
    --target_path $TARGET --drafts "${drafts[@]}" \
    --target_outputs_dir $COC \
    --val_uuids_file $VAL --test_uuids_file $TEST \
    --output_csv $csv \
    --use_3d_mrope \
    --num_draft_layers 4 --block_size 16 --mask_token_id 151662 \
    > $log 2>&1 &
  echo "GPU $gpu launched (pid=$!)"
}

run_chunk 0 "${CK_GPU0[@]}"
run_chunk 1 "${CK_GPU1[@]}"
run_chunk 2 "${CK_GPU2[@]}"
run_chunk 3 "${CK_GPU3[@]}"
sleep 1
ps -ef | grep eval_ckpt_sweep_vt | grep -v grep | wc -l

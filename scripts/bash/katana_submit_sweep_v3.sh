#!/bin/bash
# Submit the dflash hyperparameter sweep: num_draft_layers × lr grid.
# Each job runs on 4 H200 GPUs on reservation R7936711.

set -e

cd "$(dirname "$0")/.."
SCRIPT=scripts/katana_pbs_train_v3.sh

LAYERS=(1 2 3 4)
LRS=(5e-5 1e-4 2e-4)

for L in "${LAYERS[@]}"; do
  for LR in "${LRS[@]}"; do
    NAME="dflash_L${L}_lr${LR}_bs4_mrope_v3"
    echo ">>> submitting $NAME"
    qsub -v "NUM_LAYERS=$L,LR=$LR,BATCH_SIZE=4,GRAD_ACCUM=1,RUN_NAME=$NAME" \
         -N "$NAME" \
         -o "/srv/scratch/z3552416/logs/${NAME}.log" \
         "$SCRIPT"
  done
done

qstat -u $USER | tail -20

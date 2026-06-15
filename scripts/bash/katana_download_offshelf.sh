#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o /srv/scratch/z3552416/logs/offshelf_download.log
#PBS -N offshelf_dl

# Download 300 PhysicalAI-AV val-split clips (sampled with seed=42) using the
# in-package loader. CPU-only. Rsync to sharon2 afterwards for e2e eval.

set -e

export SCRATCH=/srv/scratch/z3552416
export FLORA=/srv/scratch/flora/dungda
export ALPAMAYO_OFFSHELF_DIR=$FLORA/data/alpamayo_clips_offshelf
export HF_HOME=$SCRATCH/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export N_OFFSHELF_CLIPS=300
export OFFSHELF_SEED=42

mkdir -p $ALPAMAYO_OFFSHELF_DIR $SCRATCH/logs

source $SCRATCH/envs/alpamayo/bin/activate

cd $PBS_O_WORKDIR
export PYTHONPATH=$PBS_O_WORKDIR/src:$PYTHONPATH

echo "=== starting at $(date) in $(pwd) ==="
echo "output: $ALPAMAYO_OFFSHELF_DIR"
python -V
python -c "import torch; print(f'torch {torch.__version__}')"
python -c "import physical_ai_av; print(f'physical_ai_av {physical_ai_av.__version__}')"

python scripts/cache_physical_ai_val_split.py

echo "=== finished at $(date) ==="
echo "final file count:"
ls $ALPAMAYO_OFFSHELF_DIR | wc -l
du -sh $ALPAMAYO_OFFSHELF_DIR

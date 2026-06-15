#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -o /srv/scratch/cruise/dungda/path_a/runs/cache_clips.log

SCRATCH=/srv/scratch/cruise/dungda/path_a
export HF_HOME=$SCRATCH/cache/huggingface
export HF_TOKEN=${HF_TOKEN:?must set HF_TOKEN env var}
export ALPAMAYO_CLIPS_DIR=$SCRATCH/data/alpamayo_clips
export N_CLIPS=10000

source $SCRATCH/envs/alpamayo/bin/activate
cd $SCRATCH/code/alpamayo_repo
python scripts/cache_500_clips.py

echo "=== DONE ==="
ls $SCRATCH/data/alpamayo_clips | wc -l
echo "clips saved"
du -sh $SCRATCH/data/alpamayo_clips

#!/bin/bash
#PBS -l select=1:ncpus=8:mem=64gb:ngpus=1:gpu_model=V100
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -o /srv/scratch/cruise/dungda/path_a/runs/smoke_test_dflash_v100.log

SCRATCH=/srv/scratch/cruise/dungda/path_a
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME=$SCRATCH/cache/huggingface
export VLM_PATH=$SCRATCH/models/Qwen3-VL-8B-Instruct
export PROCESSOR_PATH=$SCRATCH/models/Qwen3-VL-2B-Instruct

# Use the alpamayo uv venv (Python 3.12, torch 2.8)
source $SCRATCH/envs/alpamayo/bin/activate

# Make dflash importable
export PYTHONPATH=$SCRATCH/code/dflash:$SCRATCH/code/alpamayo_repo/src:$PYTHONPATH

cd $SCRATCH/code/alpamayo_repo
python scripts/smoke_test_dflash_draft.py

echo "=== DONE ==="

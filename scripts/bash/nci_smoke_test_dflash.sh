#!/bin/bash
#PBS -P hn98
#PBS -q gpuhopper
#PBS -l walltime=01:00:00
#PBS -l ncpus=12,ngpus=1,mem=96GB
#PBS -l storage=scratch/hn98+gdata/hn98
#PBS -l jobfs=50GB
#PBS -j oe
#PBS -o /scratch/hn98/dd9648/logs/smoke_test_dflash.log

module load cuda/12.8.0

GDATA=/g/data/hn98/dd9648
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME=$GDATA/cache/huggingface
export VLM_PATH=$GDATA/models/Qwen3-VL-8B-Instruct
export PROCESSOR_PATH=$GDATA/models/Qwen3-VL-2B-Instruct
export ALPAMAYO_CLIPS_PT=$GDATA/cache/alpamayo_example_data.pt

# Use the alpamayo uv venv (Python 3.12, has DFlash + Qwen3-VL deps)
source $GDATA/envs/alpamayo/bin/activate

# Make sure dflash package is importable.
# DFlash code lives at /g/data/hn98/dd9648/projects/dflash (cloned from local).
export PYTHONPATH=$GDATA/projects/dflash:$GDATA/projects/alpamayo_repo/src:$PYTHONPATH

cd $GDATA/projects/alpamayo_repo
python scripts/smoke_test_dflash_draft.py

echo "=== DONE ==="

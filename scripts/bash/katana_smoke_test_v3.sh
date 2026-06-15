#!/bin/bash
#PBS -q R7936711
#PBS -l select=1:ncpus=8:mem=64gb:ngpus=1:gpu_model=H200
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -o /srv/scratch/z3552416/logs/smoke_v3.log
#PBS -N smoke_v3

set -e

export SCRATCH=/srv/scratch/z3552416
export FLORA=/srv/scratch/flora/dungda
export HF_HOME=$SCRATCH/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3
export VLM_PATH=$FLORA/models/Qwen3-VL-8B-Instruct
export PROCESSOR_PATH=$FLORA/models/Qwen3-VL-2B-Instruct

source $SCRATCH/envs/alpamayo/bin/activate

cd $PBS_O_WORKDIR
export PYTHONPATH=$PBS_O_WORKDIR/src:$PYTHONPATH

echo "=== smoke test start $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devs', torch.cuda.device_count())"
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "
try:
    import flash_attn
    print('flash_attn', flash_attn.__version__, 'available')
except ImportError as e:
    print('flash_attn NOT installed:', e)
"

python scripts/smoke_test_v3.py \
    --target_path $FLORA/models/Alpamayo-R1-10B \
    --target_outputs_dir $FLORA/runs/target_coc_outputs \
    --val_uuids_file $FLORA/runs/splits/val_uuids_v3.json

echo "=== smoke test done $(date) ==="

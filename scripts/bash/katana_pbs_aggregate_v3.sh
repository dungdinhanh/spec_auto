#!/bin/bash
#PBS -l select=1:ncpus=2:mem=8gb
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -o /srv/scratch/z3552416/logs/aggregate_v3.log
#PBS -N aggregate_v3

set -e
export SCRATCH=/srv/scratch/z3552416
source $SCRATCH/envs/alpamayo/bin/activate

cd $PBS_O_WORKDIR
echo "=== aggregate v3 start $(date) ==="
python scripts/aggregate_v3_results.py \
    --runs_root /srv/scratch/flora/dungda/runs \
    --pattern 'dflash_L*_lr*_bs4_mrope_v3' \
    --output claude_report/v3_sweep_results.md
echo "=== done $(date) ==="
cat claude_report/v3_sweep_results.md

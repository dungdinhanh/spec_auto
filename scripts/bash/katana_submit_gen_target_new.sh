#!/bin/bash
# Submit 50 PBS jobs (2 L40S GPUs each = 100 shards) to generate CoC target
# outputs for the NEW clips (seed=43, ~10,476 on disk). Token IDs + pixel_values
# only (--no_logits). Resumable via skip-if-exists in the output dir.
#
# Usage: bash katana_submit_gen_target_new.sh [num_jobs]   # default 50

set -e
NUM_JOBS=${1:-50}
NUM_SHARDS=$((NUM_JOBS * 2))   # 2 shards per job (1 per GPU)
SCRATCH=/srv/scratch/cruise/dungda/path_a
OUTDIR=$SCRATCH/runs/target_coc_outputs_new
INCLUDE=$SCRATCH/runs/new_clips_to_process.json
mkdir -p $OUTDIR $SCRATCH/runs/gen_new_logs

# Per-job PBS template: 2 L40S GPUs, runs 2 shards in parallel.
SHARD_SH=$SCRATCH/runs/gen_target_new_job.sh
cat > $SHARD_SH << EOF
#!/bin/bash
#PBS -l select=1:ncpus=12:mem=96gb:ngpus=2:gpu_model=L40S
#PBS -l walltime=01:59:00
#PBS -j oe

SCRATCH=/srv/scratch/cruise/dungda/path_a
export VLM_PATH=\$SCRATCH/models/Qwen3-VL-8B-Instruct
export PROCESSOR_PATH=\$SCRATCH/models/Qwen3-VL-2B-Instruct
export HF_HOME=\$SCRATCH/cache/huggingface
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
source \$SCRATCH/envs/alpamayo/bin/activate
export PYTHONPATH=\$SCRATCH/code/dflash:\$SCRATCH/code/alpamayo_repo/src:\$PYTHONPATH
cd \$SCRATCH/code/alpamayo_repo

OUTDIR=\$SCRATCH/runs/target_coc_outputs_new
INCLUDE=\$SCRATCH/runs/new_clips_to_process.json
NUM_SHARDS=$NUM_SHARDS
S0=\$((JOB_IDX * 2))
S1=\$((JOB_IDX * 2 + 1))

CUDA_VISIBLE_DEVICES=0 python scripts/generate_target_outputs.py \
    --target_path \$SCRATCH/models/Alpamayo-R1-10B \
    --clips_dir \$SCRATCH/data/alpamayo_clips \
    --output_dir \$OUTDIR --include_uuids_file \$INCLUDE \
    --max_clips 20000 --max_new_tokens 64 \
    --num_shards \$NUM_SHARDS --shard \$S0 --no_logits > \$SCRATCH/runs/gen_new_logs/shard_\${S0}.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/generate_target_outputs.py \
    --target_path \$SCRATCH/models/Alpamayo-R1-10B \
    --clips_dir \$SCRATCH/data/alpamayo_clips \
    --output_dir \$OUTDIR --include_uuids_file \$INCLUDE \
    --max_clips 20000 --max_new_tokens 64 \
    --num_shards \$NUM_SHARDS --shard \$S1 --no_logits > \$SCRATCH/runs/gen_new_logs/shard_\${S1}.log 2>&1 &
wait
echo "=== JOB \$JOB_IDX (shards \$S0,\$S1) DONE \$(date) ==="
EOF

echo "Submitting $NUM_JOBS jobs ($NUM_SHARDS shards) ..."
for j in $(seq 0 $((NUM_JOBS - 1))); do
  JID=$(qsub -v JOB_IDX=$j \
             -o $SCRATCH/runs/gen_new_logs/job_${j}.log \
             -N gentgt_${j} \
             $SHARD_SH)
  echo "job $j: $JID"
done
echo ""
echo "Output: $OUTDIR"
echo "Monitor: qstat -u z3552416 | grep gentgt_"

#!/bin/bash
# Submit N parallel Katana jobs to cache Alpamayo clips for distillation.
#
# Usage:
#   bash katana_submit_cache_clips_parallel.sh [num_jobs] [n_clips]
#
# Example:
#   bash katana_submit_cache_clips_parallel.sh 20 10000

set -e

NUM_JOBS=${1:-20}
N_CLIPS=${2:-10000}
SCRATCH=/srv/scratch/cruise/dungda/path_a

mkdir -p $SCRATCH/runs/cache_logs

# Per-shard job script (uses env vars passed via -v)
cat > $SCRATCH/runs/cache_clips_shard.sh << EOF
#!/bin/bash
#PBS -l select=1:ncpus=2:mem=24gb
#PBS -l walltime=24:00:00
#PBS -j oe

SCRATCH=/srv/scratch/cruise/dungda/path_a
export HF_HOME=\$SCRATCH/cache/huggingface
export HF_TOKEN=\${HF_TOKEN}
export ALPAMAYO_CLIPS_DIR=\$SCRATCH/data/alpamayo_clips
export N_CLIPS=$N_CLIPS
export NUM_SHARDS=$NUM_JOBS

source \$SCRATCH/envs/alpamayo/bin/activate
cd \$SCRATCH/code/alpamayo_repo
python scripts/cache_alpamayo_clips.py
echo "=== SHARD \$SHARD_INDEX/\$NUM_SHARDS DONE ==="
EOF

# Submit N shards
for i in $(seq 0 $((NUM_JOBS - 1))); do
    JID=$(qsub -v SHARD_INDEX=$i \
               -o $SCRATCH/runs/cache_logs/shard_${i}_of_${NUM_JOBS}.log \
               -N cache_${i}_of_${NUM_JOBS} \
               $SCRATCH/runs/cache_clips_shard.sh)
    echo "Submitted shard $i: $JID"
done

echo ""
echo "All $NUM_JOBS shards submitted to cache $N_CLIPS Alpamayo clips."
echo "Output: $SCRATCH/data/alpamayo_clips/"
echo "Monitor: qstat -u z3552416"

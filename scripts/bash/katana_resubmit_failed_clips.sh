#!/bin/bash
# Resubmit the UUIDs from the seed=43 sample that failed during the first
# 100-job batch (mostly HF rate-limit failures).
#
# Approach:
#   failed_set = seed43_uuid_list  -  currently_on_disk
#
# Submit ~24 small CPU jobs (less parallelism = less HF throttling).
#
# Usage:
#   bash katana_resubmit_failed_clips.sh [num_jobs]   # default 24

set -e

NUM_JOBS=${1:-24}
SCRATCH=/srv/scratch/cruise/dungda/path_a
CLIPS_DIR=$SCRATCH/data/alpamayo_clips
SRC_UUIDS=$SCRATCH/runs/more_clips_uuids_seed43.json
RETRY_UUIDS=$SCRATCH/runs/more_clips_uuids_seed43_retry.json

mkdir -p $SCRATCH/runs/cache_logs

echo "=== computing failed UUIDs ==="
source $SCRATCH/envs/alpamayo/bin/activate
python3 - <<PY
import glob, json, os

with open("$SRC_UUIDS") as f:
    target = json.load(f)
print(f"original UUID list (seed=43): {len(target)}")

on_disk = set(
    os.path.basename(f)[:-3]
    for f in glob.glob("$CLIPS_DIR/*.pt")
)
print(f"currently on disk (all clips): {len(on_disk)}")

retry = [u for u in target if u not in on_disk]
print(f"failed (not on disk): {len(retry)}")

assert not (set(retry) & on_disk), "retry set overlaps disk!"
with open("$RETRY_UUIDS", "w") as f:
    json.dump(retry, f)
print(f"wrote retry list to $RETRY_UUIDS")
PY

# Per-shard PBS template — same as the original but smaller mem/cpu and shorter wall.
SHARD_SH=$SCRATCH/runs/cache_clips_retry_shard.sh
cat > $SHARD_SH << EOF
#!/bin/bash
#PBS -l select=1:ncpus=2:mem=16gb
#PBS -l walltime=12:00:00
#PBS -j oe

SCRATCH=/srv/scratch/cruise/dungda/path_a
export HF_HOME=\$SCRATCH/cache/huggingface
export HF_TOKEN=\${HF_TOKEN}
export ALPAMAYO_CLIPS_DIR=\$SCRATCH/data/alpamayo_clips
export UUIDS_FILE=$RETRY_UUIDS
export NUM_SHARDS=$NUM_JOBS

source \$SCRATCH/envs/alpamayo/bin/activate
cd \$SCRATCH/code/alpamayo_repo
python scripts/cache_alpamayo_clips_from_uuids.py
echo "=== RETRY SHARD \$SHARD_INDEX/\$NUM_SHARDS DONE ==="
EOF

echo ""
echo "=== submitting $NUM_JOBS retry shards ==="
for i in $(seq 0 $((NUM_JOBS - 1))); do
    JID=$(qsub -v SHARD_INDEX=$i \
               -o $SCRATCH/runs/cache_logs/retry_shard_${i}_of_${NUM_JOBS}.log \
               -N retry_${i}_of_${NUM_JOBS} \
               $SHARD_SH)
    echo "Submitted retry shard $i: $JID"
done

echo ""
echo "Submitted $NUM_JOBS retry shards."
echo "Retry UUID list: $RETRY_UUIDS"
echo "Output:          $CLIPS_DIR/"
echo "Monitor:         qstat -u z3552416 | grep retry_"

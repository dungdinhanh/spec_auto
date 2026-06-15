#!/bin/bash
# Submit N parallel Katana jobs to cache MORE Alpamayo clips, with a
# pre-computed UUID list that is DISJOINT from existing on-disk clips.
#
# Usage:
#   bash katana_submit_more_clips.sh [num_jobs] [n_sample] [seed]
#
# Defaults: 100 jobs, 11000 sampled (with seed 43), filtered against
# existing 10k → ~10,600 net-new UUIDs scattered across 100 shards.
#
# Outputs:
#   $SCRATCH/runs/more_clips_uuids_seed${SEED}.json  (UUID list)
#   $SCRATCH/runs/cache_logs/more_shard_*.log        (per-shard logs)
#   $SCRATCH/data/alpamayo_clips/<uuid>.pt           (new clips)

set -e

NUM_JOBS=${1:-100}
N_SAMPLE=${2:-11000}
SEED=${3:-43}
SCRATCH=/srv/scratch/cruise/dungda/path_a
CLIPS_DIR=$SCRATCH/data/alpamayo_clips
UUIDS_FILE=$SCRATCH/runs/more_clips_uuids_seed${SEED}.json

mkdir -p $SCRATCH/runs/cache_logs

# Step 1: build disjoint UUID list (this is fast — only file stats + a single
# random.sample on the index, no HF traffic).
echo "=== building disjoint UUID list (seed=$SEED, sample=$N_SAMPLE) ==="
source $SCRATCH/envs/alpamayo/bin/activate
python3 - <<PY
import glob, json, os, random
import physical_ai_av

avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
all_ids = avdi.clip_index.index.tolist()
print(f"total dataset clips: {len(all_ids)}")

random.seed($SEED)
sample = random.sample(all_ids, $N_SAMPLE)
existing = set(
    os.path.basename(f)[:-3]
    for f in glob.glob("$CLIPS_DIR/*.pt")
)
print(f"existing on disk: {len(existing)}")

to_download = [c for c in sample if c not in existing]
print(f"sample size: {len(sample)}, after filtering existing: {len(to_download)}")

# Verify zero overlap.
assert not (set(to_download) & existing), "UUID list overlaps with existing!"

with open("$UUIDS_FILE", "w") as f:
    json.dump(to_download, f)
print(f"wrote {len(to_download)} UUIDs to $UUIDS_FILE")
PY

# Step 2: write per-shard PBS template.
SHARD_SH=$SCRATCH/runs/cache_clips_more_shard.sh
cat > $SHARD_SH << EOF
#!/bin/bash
#PBS -l select=1:ncpus=2:mem=24gb
#PBS -l walltime=24:00:00
#PBS -j oe

SCRATCH=/srv/scratch/cruise/dungda/path_a
export HF_HOME=\$SCRATCH/cache/huggingface
export HF_TOKEN=\${HF_TOKEN}
export ALPAMAYO_CLIPS_DIR=\$SCRATCH/data/alpamayo_clips
export UUIDS_FILE=$UUIDS_FILE
export NUM_SHARDS=$NUM_JOBS

source \$SCRATCH/envs/alpamayo/bin/activate
cd \$SCRATCH/code/alpamayo_repo
python scripts/cache_alpamayo_clips_from_uuids.py
echo "=== SHARD \$SHARD_INDEX/\$NUM_SHARDS DONE seed=$SEED ==="
EOF

# Step 3: submit N shards.
echo ""
echo "=== submitting $NUM_JOBS shards ==="
for i in $(seq 0 $((NUM_JOBS - 1))); do
    JID=$(qsub -v SHARD_INDEX=$i \
               -o $SCRATCH/runs/cache_logs/more_shard_${i}_of_${NUM_JOBS}_seed${SEED}.log \
               -N more_${i}_of_${NUM_JOBS} \
               $SHARD_SH)
    echo "Submitted shard $i: $JID"
done

echo ""
echo "Submitted $NUM_JOBS shards to cache new Alpamayo clips."
echo "UUID list: $UUIDS_FILE"
echo "Output:    $CLIPS_DIR/"
echo "Monitor:   qstat -u z3552416 | grep more_"

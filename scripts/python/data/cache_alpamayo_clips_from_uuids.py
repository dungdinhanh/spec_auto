"""Cache Alpamayo PhysicalAI clips for a precomputed list of UUIDs.

Replaces the random.sample logic of cache_alpamayo_clips.py with a UUID file
loaded from `UUIDS_FILE`. This guarantees ZERO OVERLAP with any preexisting
set of clips (the filtering against `OUTPUT_DIR` happens at UUID-file
construction time, see `katana_submit_more_clips.sh`).

Env vars:
  UUIDS_FILE           path to a JSON list of UUIDs (required)
  SHARD_INDEX          0-based shard index
  NUM_SHARDS           total parallel workers
  ALPAMAYO_CLIPS_DIR   output directory

Per-shard work: uuids[SHARD_INDEX::NUM_SHARDS].
"""
import gc
import json
import os
import random
import time

import torch
import physical_ai_av  # noqa: F401  (needed for side-effect: registers loaders)
from huggingface_hub.errors import HfHubHTTPError

from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper

UUIDS_FILE = os.environ["UUIDS_FILE"]
OUTPUT_DIR = os.environ.get(
    "ALPAMAYO_CLIPS_DIR", "/srv/scratch/cruise/dungda/path_a/data/alpamayo_clips"
)
SHARD_INDEX = int(os.environ.get("SHARD_INDEX", "0"))
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "1"))

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(UUIDS_FILE) as f:
    all_uuids = json.load(f)
print(f"Loaded {len(all_uuids)} UUIDs from {UUIDS_FILE}", flush=True)

my_clips = all_uuids[SHARD_INDEX::NUM_SHARDS]
print(
    f"shard {SHARD_INDEX}/{NUM_SHARDS} -> {len(my_clips)} clips",
    flush=True,
)


def load_with_retry(clip_id, max_retries=8):
    delay = 5.0
    for attempt in range(max_retries):
        try:
            return load_physical_aiavdataset(clip_id, t0_us=5_100_000)
        except HfHubHTTPError as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = delay * (2 ** attempt) + random.uniform(0, 5)
                print(
                    f"  rate-limited on {clip_id}, retry {attempt+1}/{max_retries} in {wait:.0f}s",
                    flush=True,
                )
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"max retries exceeded for {clip_id}")


saved = 0
skipped = 0
failed = 0
total = len(my_clips)
# Stagger workers to avoid simultaneous initial bursts.
time.sleep(SHARD_INDEX * 3)
for i, clip_id in enumerate(my_clips):
    out_path = os.path.join(OUTPUT_DIR, f"{clip_id}.pt")
    if os.path.exists(out_path):
        # Should be rare with the disjoint UUID list — log if it happens.
        skipped += 1
        continue
    try:
        data = load_with_retry(clip_id)
        messages = helper.create_message(data["image_frames"].flatten(0, 1))
        torch.save({"data": data, "messages": messages, "clip_id": clip_id}, out_path)
        saved += 1
        del data, messages
        gc.collect()
    except Exception as e:
        failed += 1
        print(f"  skip {clip_id}: {type(e).__name__}: {e}", flush=True)

    if (i + 1) % 25 == 0:
        print(
            f"  shard {SHARD_INDEX}/{NUM_SHARDS} progress {i+1}/{total}  "
            f"saved={saved} skipped={skipped} failed={failed}",
            flush=True,
        )

print(
    f"\nDone. shard={SHARD_INDEX}/{NUM_SHARDS}, "
    f"saved={saved}, skipped(unexpected)={skipped}, failed={failed}, dir={OUTPUT_DIR}"
)
print("DONE")

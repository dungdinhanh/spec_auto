"""Cache N random Alpamayo PhysicalAI clips for distillation training.

Saves each clip incrementally to its own .pt file under OUTPUT_DIR so a long
download run won't OOM and is resumable on restart. Supports parallel sharding.

Configurable via env vars:
  N_CLIPS              total number of clips to sample (default 500)
  SEED                 random seed for sampling (default 42)
  SHARD_INDEX          0-based index of this worker (default 0)
  NUM_SHARDS           total number of parallel workers (default 1)
  ALPAMAYO_CLIPS_DIR   output directory

Output:
  $ALPAMAYO_CLIPS_DIR/<clip_id>.pt
"""
import os
import gc
import random
import time
import torch
import physical_ai_av
from huggingface_hub.errors import HfHubHTTPError

from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper

OUTPUT_DIR = os.environ.get(
    "ALPAMAYO_CLIPS_DIR", "/srv/scratch/z3552416/path_a/data/alpamayo_clips"
)
N_CLIPS = int(os.environ.get("N_CLIPS", "500"))
SEED = int(os.environ.get("SEED", "42"))
SHARD_INDEX = int(os.environ.get("SHARD_INDEX", "0"))   # 0-based
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "1"))

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading clip index from PhysicalAI dataset...", flush=True)
avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
all_ids = avdi.clip_index.index.tolist()
print(f"Total clips available: {len(all_ids)}", flush=True)

random.seed(SEED)
sampled = random.sample(all_ids, N_CLIPS)
# Shard the sampled list across workers (each worker takes every NUM_SHARDS-th id)
my_clips = sampled[SHARD_INDEX::NUM_SHARDS]
print(
    f"Sampled {N_CLIPS} clip ids, shard {SHARD_INDEX}/{NUM_SHARDS} -> {len(my_clips)} clips",
    flush=True,
)

def load_with_retry(clip_id, max_retries=8):
    """Load a clip, retrying on HF rate-limit errors with exponential backoff."""
    delay = 5.0
    for attempt in range(max_retries):
        try:
            return load_physical_aiavdataset(clip_id, t0_us=5_100_000)
        except HfHubHTTPError as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = delay * (2 ** attempt) + random.uniform(0, 5)
                print(f"  rate-limited on {clip_id}, retry {attempt+1}/{max_retries} in {wait:.0f}s", flush=True)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"max retries exceeded for {clip_id}")


saved = 0
skipped = 0
failed = 0
total = len(my_clips)
# Stagger workers to avoid simultaneous initial bursts
time.sleep(SHARD_INDEX * 3)
for i, clip_id in enumerate(my_clips):
    out_path = os.path.join(OUTPUT_DIR, f"{clip_id}.pt")
    if os.path.exists(out_path):
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
        print(f"  shard {SHARD_INDEX}/{NUM_SHARDS} progress {i+1}/{total}  saved={saved} skipped={skipped} failed={failed}", flush=True)

print(f"\nDone. shard={SHARD_INDEX}/{NUM_SHARDS}, saved={saved}, skipped(already exist)={skipped}, dir={OUTPUT_DIR}")
print("DONE")

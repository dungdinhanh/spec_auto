"""Download 300 PhysicalAI-AV val-split clips using the in-package loader
`alpamayo_r1.load_physical_aiavdataset.load_physical_aiavdataset` (which is
compatible with physical_ai_av 0.2.x). Produces the same per-clip schema as
`cache_alpamayo_clips.py` (data + clip_id; messages built later on sharon2).

Sampling:
  - sample 300 UUIDs from clip_index[split=='val'] with fixed seed=42
  - saves the sampled UUID list to test_offshelf_uuids.json

Environment variables:
  ALPAMAYO_OFFSHELF_DIR   output dir
  N_OFFSHELF_CLIPS        total to sample (default 300)
  OFFSHELF_SEED           random seed (default 42)
  SHARD_INDEX / NUM_SHARDS   optional sharding (default 0/1)
"""
from __future__ import annotations

import gc
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import physical_ai_av

try:
    from huggingface_hub.errors import HfHubHTTPError
except ImportError:
    class HfHubHTTPError(Exception): ...

# In-package loader that uses physical_ai_av 0.2.x API under the hood.
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset


OUTPUT_DIR = os.environ.get(
    "ALPAMAYO_OFFSHELF_DIR",
    str(Path.cwd() / "alpamayo_clips_offshelf"),
)
N_CLIPS = int(os.environ.get("N_OFFSHELF_CLIPS", "300"))
SEED = int(os.environ.get("OFFSHELF_SEED", "42"))
SHARD_INDEX = int(os.environ.get("SHARD_INDEX", "0"))
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "1"))


def load_with_retry(clip_id: str, avdi, max_retries: int = 8):
    delay = 5.0
    for attempt in range(max_retries):
        try:
            return load_physical_aiavdataset(clip_id, t0_us=5_100_000, avdi=avdi)
        except HfHubHTTPError as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = delay * (2 ** attempt) + random.uniform(0, 5)
                print(f"  rate-limited on {clip_id}, retry {attempt+1}/{max_retries} in {wait:.0f}s",
                      flush=True)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"max retries exceeded for {clip_id}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output dir: {OUTPUT_DIR}", flush=True)
    print(f"Shard {SHARD_INDEX}/{NUM_SHARDS}, target={N_CLIPS} clips, seed={SEED}", flush=True)

    print("Loading PhysicalAI-AV clip index ...", flush=True)
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    avdi.download_metadata()
    idx = avdi.clip_index
    if "split" not in idx.columns:
        raise RuntimeError(f"No 'split' column in clip_index; columns={list(idx.columns)}")
    val_ids = idx[idx["split"] == "val"].index.tolist()
    print(f"val-split size: {len(val_ids)}", flush=True)

    rng = random.Random(SEED)
    sampled = sorted(rng.sample(val_ids, min(N_CLIPS, len(val_ids))))

    uuids_path = os.path.join(OUTPUT_DIR, "test_offshelf_uuids.json")
    with open(uuids_path, "w") as f:
        json.dump(sampled, f, indent=2)
    print(f"Wrote sampled UUID list ({len(sampled)}) -> {uuids_path}", flush=True)

    my_clips = sampled[SHARD_INDEX::NUM_SHARDS]
    print(f"This shard will download {len(my_clips)} clips", flush=True)

    saved, skipped, failed = 0, 0, 0
    t0 = time.time()
    for i, clip_id in enumerate(my_clips):
        out_path = os.path.join(OUTPUT_DIR, f"{clip_id}.pt")
        if os.path.exists(out_path):
            skipped += 1
            continue
        try:
            data = load_with_retry(clip_id, avdi)
            torch.save({"data": data, "clip_id": clip_id}, out_path)
            saved += 1
            del data
            gc.collect()
        except Exception as e:
            failed += 1
            print(f"  skip {clip_id}: {type(e).__name__}: {str(e)[:140]}", flush=True)

        if (i + 1) % 10 == 0:
            elapsed = max(time.time() - t0, 1e-9)
            rate = saved / elapsed
            eta_s = (len(my_clips) - (i + 1)) / max(rate, 1e-9)
            print(f"  [{i+1}/{len(my_clips)}] saved={saved} skipped={skipped} "
                  f"failed={failed} rate={rate:.2f} clips/s  eta={eta_s/60:.1f}min",
                  flush=True)

    print(f"\nDone. shard={SHARD_INDEX}/{NUM_SHARDS}, saved={saved}, "
          f"skipped={skipped}, failed={failed}", flush=True)


if __name__ == "__main__":
    main()

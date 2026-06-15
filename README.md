<div align="center">

# Reasoning Aware Speculative Decoding for VLAs in Autonomous Driving

</div>

> **Note for reviewers.** This repository is the anonymous code release for our speculative-decoding work on Alpamayo-R1. It forks [NVlabs/alpamayo](https://github.com/NVlabs/alpamayo) and adds DFlash + EAGLE-3 draft models, AARL post-training, target-output generation, training/eval pipelines, and launch scripts for multi-cluster training. See **[Speculative Decoding for Alpamayo-R1](#speculative-decoding-for-alpamayo-r1)** below.

<div align="center">

[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Alpamayo--R1--10B-blue)](https://huggingface.co/nvidia/Alpamayo-R1-10B)
[![arXiv](https://img.shields.io/badge/arXiv-2511.00088-b31b1b.svg)](https://arxiv.org/abs/2511.00088)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](./LICENSE)

</div>

## Updates

- [April 2026] ⚙️ [Fine-tuning scripts](#fine-tuning-scripts) released: [SFT](docs/FINETUNE_SFT.md) for supervised fine-tuning and [RL](finetune/rl/README.md) for reinforcement learning-based post-training.
- [March 2026] [🏔️ Alpamayo 1.5](https://github.com/NVlabs/alpamayo1.5) has been released! We recommend all users check out the new version for improved performance, new features, and continued support! 🚀
- [January 2026] Following the release of [NVIDIA Alpamayo](https://nvidianews.nvidia.com/news/alpamayo-autonomous-vehicle-development) at CES 2026, Alpamayo-R1 has been renamed to Alpamayo 1.

______________________________________________________________________

**📖 Please read the [HuggingFace Model Card](https://huggingface.co/nvidia/Alpamayo-R1-10B) first!**
The model card contains comprehensive details on model architecture, inputs/outputs, licensing, and tested hardware configurations. This GitHub README focuses on setup, usage, and frequently asked questions.

## Requirements

| Requirement | Specification                                                       |
| ----------- | ------------------------------------------------------------------- |
| **Python**  | 3.12.x (see `pyproject.toml`)                                       |
| **GPU**     | NVIDIA GPU with ≥24 GB VRAM (e.g., RTX 3090, RTX 4090, A5000, H100) |
| **OS**      | Linux (tested); other platforms unverified                          |

> ⚠️ **Note**: GPUs with less than 24 GB VRAM will likely encounter CUDA out-of-memory errors.

## Installation

### 1. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

### 2. Set up the environment

```bash
uv venv ar1_venv
source ar1_venv/bin/activate
uv sync --active
```

### 3. Authenticate with HuggingFace

The model requires access to gated resources. Request access here:

- 🤗 [Physical AI AV Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
- 🤗 [Alpamayo Model Weights](https://huggingface.co/nvidia/Alpamayo-R1-10B)

Then authenticate using the HuggingFace CLI:

```bash
pip install -U huggingface_hub
hf auth login
```

Get your access token at: https://huggingface.co/settings/tokens

> 💡 **Tip**: For more details on HuggingFace authentication, see the [official documentation](https://huggingface.co/docs/huggingface_hub/guides/cli).

## Running Inference

### Test script

NOTE: This script will download both some example data (relatively small) and the model weights (22 GB).
The latter can be particularly slow depending on network bandwidth.
For reference, it takes around 2.5 minutes on a 100 MB/s wired connection.

```bash
python src/alpamayo_r1/test_inference.py
```

In case you would like to obtain more trajectories and reasoning traces, please feel free to change
the `num_traj_samples=1` argument to a higher number (Line 60).

### Interactive notebook

We provide a notebook with similar inference code at `notebook/inference.ipynb`.

## Relationship with the Paper

Alpamayo 1 implements the architecture described in our paper [*"Alpamayo-R1: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail
"*](https://arxiv.org/abs/2511.00088), including:

| Feature                                 | Paper Description                                                | This Release (v1.0)    |
| --------------------------------------- | ---------------------------------------------------------------- | ---------------------- |
| **Chain-of-Causation (CoC) reasoning**  | Hybrid auto-labeling with human in the loop for reasoning traces | ✅ Included            |
| **Vision-Language-Action architecture** | Cosmos-Reason backbone + action expert                           | ✅ Included            |
| **Trajectory prediction**               | 6.4s horizon, 64 waypoints at 10 Hz                              | ✅ Included            |
| **SFT fine-tuning (weights)**           | SFT trained model weights                                        | ✅ Included            |
| **SFT fine-tuning (code)**              | Supervised fine-tuning pipeline                                  | ✅ Included            |
| **RL post-training (weights)**          | RL post-trained model weights                                    | ❌ Not in this release |
| **RL post-training (code)**             | RL post-training pipeline via Cosmos-RL                          | ✅ Included            |
| **Route/navigation conditioning**       | Explicit navigation or route inputs                              | ❌ Not in this release |
| **Meta-actions/General VQA**            | High-level behavior and visual question answering                | ❌ Not in this release |

This release includes the core model, SFT scripts, and the RL post-training pipeline. RL-trained weights, route conditioning, and meta-actions are candidates for future releases.

## Fine-tuning Scripts

| Method  | Description                                              | Docs                              |
| ------- | -------------------------------------------------------- | --------------------------------- |
| **SFT** | Supervised fine-tuning                                   | [SFT guide](docs/FINETUNE_SFT.md) |
| **RL**  | Reinforcement learning-based post-training via Cosmos-RL | [RL guide](finetune/rl/README.md) |

Please refer to the linked guides for compute requirements, step-by-step
instructions, and fine-tuning FAQ.

## Speculative Decoding for Alpamayo-R1

This fork adds a speculative-decoding pipeline that accelerates Alpamayo-R1's
chain-of-causation (CoC) reasoning. The autoregressive VLM stream produces
~15–20 reasoning tokens per scene before the diffusion head emits a trajectory,
and that AR phase is what speculative decoding speeds up.

### What's in this fork

| Component | Location | What it does |
|---|---|---|
| **DFlash draft** (block-diffusion) | `src/dflash/`, `src/alpamayo_r1/models/dflash_draft.py` (1D RoPE), `dflash_draft_mrope.py` (3D M-RoPE) | Block-diffusion draft model: predicts `block_size − 1` tokens in parallel by denoising a masked block conditioned on target hidden states from multiple layers. |
| **EAGLE-3 draft** (chain decoding) | `src/alpamayo_r1/models/eagle3_draft.py`, `eagle3_infer.py` | Paper-faithful EAGLE-3 chain draft (single decoder layer + LM head + fc projection over 3 target layers), with 1D and 3D rotary variants. |
| **AR draft baseline** | `src/alpamayo_r1/models/autoregressive_draft.py` | Small autoregressive draft for baseline comparison. |
| **DFlash SFT trainer** | `scripts/python/training/train_dflash_distillation_v7.py` | Block-diffusion supervised fine-tuning with weighted CE + optional KL distillation against the target's full token distribution. |
| **DFlash AARL trainer** | `scripts/python/training/train_dflash_rl_action_v5.py` | Multi-block contamination AARL: K=32 rollouts per rejection block, GRPO advantage, KL anchor against a frozen ref draft, action-MSE + token-match reward. |
| **EAGLE-3 SFT trainer** | `scripts/python/training/train_eagle3.py` | EAGLE-3 chain SFT (rollout-length 7, target_layer_ids [1, 17, 32]). |
| **EAGLE-3 AARL trainer** | `scripts/python/training/train_eagle3_rl_action_v2.py` | Multi-block contamination port of the DFlash AARL to chain decoding. |
| **Target-output generation** | `scripts/python/training/generate_target_outputs.py` | Runs Alpamayo's greedy CoC + caches `prompt_input_ids`, optional `output_logits`, `pixel_values`, `image_grid_thw` per clip — the format consumed by the SFT and AARL trainers. |
| **End-to-end spec-decode eval** | `scripts/python/evaluation/benchmark_spec_decoding.py`, plus per-arch e2e scripts | Measures L (avg accepted tokens per iter), c-ratio (draft cost / verify cost), and wall-clock speedup vs AR baseline. |

Launch scripts under `scripts/bash/` are concrete examples used during our
research; they bake in cluster-specific paths (sharon = bare-metal H100 NVL,
katana = UNSW HPC with H200 reservation + L40S) and serve as templates for
adapting to your own environment.

### Environment setup

Follow the upstream Alpamayo install ([Installation](#installation)), then make
the additional draft / DFlash code importable:

```bash
# 1. Upstream Alpamayo env (uv venv ar1_venv + uv sync --active per upstream)
source ar1_venv/bin/activate

# 2. Put this fork's src/ and the DFlash submodule on PYTHONPATH
export PYTHONPATH="$PWD/src:$PWD/src/dflash:${PYTHONPATH:-}"

# 3. Model paths (used by the training/eval scripts)
export TARGET_PATH=/path/to/Alpamayo-R1-10B            # target VLM (10B)
export VLM_PATH=/path/to/Qwen3-VL-8B-Instruct          # for building drafts
export PROCESSOR_PATH=/path/to/Qwen3-VL-2B-Instruct    # tokenizer / processor
```

The trainers use `torch.distributed` via `torchrun`. K=32 + multi-block AARL fits
8× 94GB H100 NVL with `--k_chunk_size 4`. SFT runs comfortably on the same
hardware with `batch_size=2 --grad_accum_steps=2`.

### Data preparation

The pipeline needs three artifacts before training: (a) raw Alpamayo driving
clips (.pt files per UUID), (b) the val/test UUID split, and (c) cached target
outputs (greedy CoC tokens) per clip.

#### 1. Get the raw clips

The original Alpamayo clips ship with the upstream `nvidia/PhysicalAI-Autonomous-Vehicles`
dataset (gated — request access first, then `hf auth login`).

```bash
# Cache a subset of clips locally (example: download 22k clips by UUID)
python scripts/python/training/cache_alpamayo_clips.py \
    --output_dir /path/to/alpamayo_clips \
    --num_clips 22000 \
    --hf_dataset nvidia/PhysicalAI-Autonomous-Vehicles
```

We also provide `cache_physical_ai_val_split.py` (for the off-shelf val split)
and `cache_alpamayo_clips_from_uuids.py` (when you already have a UUID list).

#### 2. Define val/test splits

Splits are JSON arrays of clip UUIDs. Example (`splits_v3/`):

```bash
# val_uuids_v3.json   — 300 UUIDs
# test_uuids_v3.json  — 200 UUIDs
# (train = everything else, set automatically by excluding val/test)
```

The repo doesn't ship splits; pick UUIDs from your local clip pool.

#### 3. Generate target CoC outputs

Each training clip needs the target model's greedy CoC tokens cached. This is
the one-time expensive step (~hours on 8 GPUs for 22k clips).

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/python/training/generate_target_outputs.py \
    --target_path $TARGET_PATH \
    --clips_dir /path/to/alpamayo_clips \
    --output_dir /path/to/target_coc_outputs_all \
    --max_clips 22000 \
    --no_logits  # set false to also cache full token distributions for KL distillation
```

Output files are named `<uuid>.pt` and contain `prompt_input_ids`,
`output_token_ids`, `pixel_values`, `image_grid_thw`, and optionally
`output_logits`.

### Training pipeline

The full pipeline is **SFT → AARL** on top of the SFT init.

#### Stage 1 — SFT (block-diffusion CE, ~30 epochs)

```bash
torchrun --nproc_per_node=8 scripts/python/training/train_dflash_distillation_v7.py \
    --target_path $TARGET_PATH \
    --target_outputs_dir /path/to/target_coc_outputs_all \
    --val_uuids_file  splits/val_uuids_v3.json \
    --test_uuids_file splits/test_uuids_v3.json \
    --output_dir runs/dflash_L2_22k_sft_v7 \
    --num_draft_layers 2 --block_size 16 \
    --num_target_features 5 --target_layer_ids 1,9,17,25,33 \
    --num_epochs 30 --lr 1e-4 \
    --batch_size 2 --grad_accum_steps 2 \
    --overlapping_blocks --random_mask
```

For EAGLE-3 SFT use `train_eagle3.py` with `--rollout_length 7 --target_layer_ids 1,17,32`.
3D rotary variants add `--use_mrope3d_draft`. Concrete examples:
`scripts/bash/katana_train_dflash_20k_L2L4.sh`, `katana_train_dflash_20k_3d_L2L4.sh`,
`katana_train_eagle3_22k_1d_3d.sh`.

#### Stage 2 — AARL post-training (multi-block contamination, 1 epoch)

```bash
torchrun --nproc_per_node=8 scripts/python/training/train_dflash_rl_action_v5.py \
    --target_path $TARGET_PATH \
    --init_draft_path runs/dflash_L2_22k_sft_v7/draft_final.pt \
    --target_outputs_dir /path/to/target_coc_outputs_all \
    --val_uuids_file  splits/val_uuids_v3.json \
    --test_uuids_file splits/test_uuids_v3.json \
    --output_dir runs/dflash_L2_22k_aarl_v5 \
    --num_target_features 5 \
    --num_epochs 1 --lr 1e-6 --kl_weight 0.02 \
    --k_samples 32 --k_chunk_size 4 --temperature 1.0 \
    --w_traj 1.0 --w_cons 0.0 --w_text 0.5 \
    --multiblock_N 5 --multiblock_max_total 20 \
    --anchor_source ref --filter_to_rejection_blocks
```

For EAGLE-3 use `train_eagle3_rl_action_v2.py` with `--block_size 8` (γ=7) and the
same multi-block / K=32 settings (best lr=1e-5 for EAGLE-3 vs 1e-6 for DFlash —
the optimal lr is architecture-dependent).

Key arguments worth remembering:
- `--multiblock_N` — window size of consecutive positions to contaminate per rejection block. 0 = legacy single-block; ≥1 enables multi-block iteration.
- `--multiblock_max_total` — cumulative contaminated-position budget per step.
- `--filter_to_rejection_blocks` — per-step probe of all `block_start` values; skips clips with zero rejections.
- `--anchor_source ref` — anchor KL against the FROZEN ref draft (not the live policy). Required for stability.
- `--k_samples 32 --k_chunk_size 4` — K=32 GRPO rollouts, chunked through the target VLM 4 at a time.

#### AARL early stopping

AARL ckpts tend to peak mid-training and decline. Save every 500 steps and
evaluate intermediate ckpts (`draft_step_500.pt`, `draft_step_1000.pt`, ...) on
val/test — final is often not best.

### Evaluation

End-to-end speculative-decode benchmark — measures L (avg accepted tokens / iter),
c (draft cost / verify cost), and wall-clock speedup against AR baseline:

```bash
# DFlash draft
CUDA_VISIBLE_DEVICES=0 python scripts/python/evaluation/benchmark_spec_decoding.py \
    --target_path $TARGET_PATH \
    --draft_path runs/dflash_L2_22k_aarl_v5/draft_final.pt \
    --clips_dir /path/to/target_coc_outputs_all \
    --uuids_file splits/val_uuids_v3.json \
    --num_draft_layers 2 --block_size 16 --num_target_features 5 \
    --output_json eval/dflash_val.json

# EAGLE-3 draft  — use the matching e2e script (1D vs 3D)
# scripts/python/evaluation/bench_eagle3_timings.py
```

Other useful eval utilities:

- `scripts/python/evaluation/test_draft_accuracy.py` — per-block argmax-match accuracy (no end-to-end timing).
- `scripts/python/evaluation/bench_dflash_c_ratio.py` — measures `c = draft_cost / verify_cost`.
- `scripts/python/evaluation/eval_draft_output_region.py` — accuracy specifically on the output region (skips the long prompt prefix).

### Reproducing the headline result

Best-of-sweep recipe, in one line each:

| Architecture | Init | Recipe | Test L (val L) | Δ vs SFT init |
|---|---|---|---|---|
| EAGLE-3 1D + AARL | 22k SFT 1D | `train_eagle3_rl_action_v2.py --multiblock_N 5 --multiblock_max_total 10 --lr 1e-5 --kl_weight 0.02 --k_samples 32` | **5.34** (5.51) | +0.067 |
| DFlash L=2 1D + AARL | 22k SFT 1D | `train_dflash_rl_action_v5.py --multiblock_N 5 --multiblock_max_total 20 --lr 1e-6 --kl_weight 0.02 --k_samples 32` | 4.75 (5.00) | +0.014 |

The `archive/all-versions` branch on github contains every intermediate trainer
version (v2-v4 DFlash AARL, v2-v6 DFlash SFT, v1 EAGLE-3 AARL, etc.) for
reference if you want to trace the development history.

## Project Structure

```
alpamayo/
├── finetune/
│   ├── rl/                              # RL post-training
│   │   ├── models/                      # Model wrappers & Cosmos-RL entry scripts
│   │   ├── rewards/                     # Reward functions
│   │   ├── prefetch/                    # Shared-memory data prefetch server
│   │   ├── toml/                        # Cosmos-RL training configs
│   │   ├── hydra_configs/               # Dataset & preprocessing configs
│   │   └── README.md                    # RL post-training guide
│   └── sft/                             # Supervised fine-tuning
│       ├── configs/                     # Model configs
│       ├── models/                      # Trainable wrappers
│       ├── train_hf.py                  # Training script
│       └── evaluate_hf.py               # Evaluation script
├── notebook/
│   └── inference.ipynb                  # Example notebook
├── src/
│   └── alpamayo_r1/
│       ├── action_space/
│       │   └── ...                      # Action space definitions
│       ├── diffusion/
│       │   └── ...                      # Diffusion model components
│       ├── geometry/
│       │   └── ...                      # Geometry utilities and modules
│       ├── models/
│       │   ├── ...                      # Model components and utils functions
│       ├── __init__.py                  # Package marker
│       ├── config.py                    # Model and experiment configuration
│       ├── helper.py                    # Utility functions
│       ├── load_physical_aiavdataset.py # Dataset loader
│       ├── test_inference.py            # Inference test script
├── pyproject.toml                       # Project dependencies
└── uv.lock                              # Locked dependency versions
```

## Troubleshooting

### Flash Attention issues

The model uses Flash Attention 2 by default. If you encounter compatibility issues:

```python
# Use PyTorch's scaled dot-product attention instead
config.attn_implementation = "sdpa"
```

### CUDA out-of-memory errors

If you encounter OOM errors:

1. Ensure you have a GPU with at least 24 GB VRAM
2. Reduce `num_traj_samples` if generating multiple trajectories
3. Close other GPU-intensive applications

## License

- **Inference code**: Apache License 2.0 - see [LICENSE](./LICENSE) for details.
- **Model weights**: Non-commercial license - see [HuggingFace Model Card](https://huggingface.co/nvidia/Alpamayo-R1-10B) for details.

## Disclaimer

Alpamayo 1 is a pre-trained reasoning model designed to accelerate research and development in the autonomous vehicle (AV) domain. It is intended to serve as a foundation for a range of AV-related use cases-from instantiating an end-to-end backbone for autonomous driving to enabling reasoning-based auto-labeling tools. In short, it should be viewed as a building block for developing customized AV applications.

Important notes:

- Alpamayo 1 is provided solely for research, experimentation, and evaluation purposes.
- Alpamayo 1 is not a fully fledged driving stack. Among other limitations, it lacks access to critical real-world sensor inputs, does not incorporate required diverse and redundant safety mechanisms, and has not undergone automotive-grade validation for deployment.

By using this model, you acknowledge that it is a research tool intended to support scientific inquiry, benchmarking, and exploration—not a substitute for a certified AV stack. The developers and contributors disclaim any responsibility or liability for the use of the model or its outputs.


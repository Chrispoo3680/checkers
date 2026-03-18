# Checkers

Checkers is a deep learning chess engine project built around a single neural network with two heads:

- Policy head: predicts strong candidate moves.
- Value head: predicts the evaluation of the current position.

The long-term goal is to train a practical, strong chess model that combines large-scale engine-supervised learning (Stockfish and Lichess evals), tactical sharpening (Lichess puzzles), and human-style priors (Magnus Carlsen games), then uses MCTS at inference time for stronger move selection.

## Main Goal

This project aims to answer a practical research question:

How far can a custom AlphaZero-style architecture and training pipeline be pushed using only curated public chess datasets and efficient training code?

Concretely, Checkers focuses on:

- Learning position understanding from massive FEN + eval corpora.
- Learning strong move priors from both engine PV lines and human games.
- Combining policy and value outputs with Monte Carlo Tree Search.
- Iterating architecture versions and measuring training outcomes across runs.

## Current Project State

The codebase already contains:

- Multiple model generations (`CheckersNetV1`, `CheckersNetV2`, `CheckersNetV3`).
- End-to-end training with optional single-GPU and DDP execution.
- Cached preprocessing pipeline for very large datasets (Parquet cache generation).
- Data processing for all core datasets currently in use.
- MCTS inference implementation and terminal gameplay script.
- Historical checkpoints and results across major iterations (`checkers1.x` to `checkers4.x`).

### Notable Progress

From the saved results files, the best run by final `test_loss` is currently:

- `checkers3.3_results.json`: `test_loss=1044.303`, `value_score=0.8836`, `policy_top1=0.3198`, `policy_top3=0.5934`.

Recent 4.x runs operate in the newer 8x8x73 policy space and are not directly comparable one-to-one with early runs that used different output conventions and training settings.

## Model Overview

### Input Representation

Positions are encoded to an 18x8x8 tensor:

- 12 piece planes (white + black pieces)
- 1 side-to-move plane
- 4 castling-right planes
- 1 en-passant plane

### Output Representation

Current policy encoding uses AlphaZero-style 8x8x73 planes:

- 56 sliding move planes (8 directions x 1..7 squares)
- 8 knight move planes
- 9 underpromotion planes (N/B/R x left/forward/right)

Policy size = 8 x 8 x 73.

### Current Main Network

`CheckersNetV3` in `src/model/models.py` uses:

- Initial conv stem
- 15 residual trunk blocks at 256 channels
- periodic SE residual blocks for channel attention
- Conv-only policy head producing 8x8x73 logits
- Value head with global average pooling and MLP to a tanh-bounded scalar

## Datasets Used

Checkers is explicitly trained from these sources:

- `magnus_carlsen_games` (human high-level game trajectories)
- `stockfish_position_evaluations` (top-5 moves + evals)
- `lichess_puzzles` (tactical sequences)
- `lichess_db_eval` (large-scale engine eval data)

Dataset settings and download metadata live in `config.yaml`.

## Training Pipeline

The pipeline is designed for large-scale data and repeatable experiments:

1. Dataset selection and discovery.
2. Dataset-specific processing into canonical columns (`fen`, `moves`, `evaluation`, optional `move_weights`).
3. One-time conversion to compact cached Parquet training shards with pre-encoded tensors.
4. Chunked lazy loading and LRU chunk caching in the dataset class.
5. Block-based distributed sampling for better sequential locality over large cached files.
6. Mixed precision (where hardware supports it), gradient clipping, cosine LR schedule, early stopping.

## Evaluation and Inference

At inference time, the model can be used directly or through MCTS:

- `src/evaluate/mcts.py` runs AlphaZero-style PUCT search.
- Legal move masking is applied in policy space before softmax.
- Terminal game loop is available via `play_against_ai_terminal.py`.

## Project Structure Map

### Root

- `train.py`: main training entrypoint, CLI, dataset selection, launch strategy (single process or DDP), checkpoint/result writing.
- `generate_stockfish_dataset.py`: generates supervised Stockfish move/eval dataset from source FEN positions.
- `play_against_ai_terminal.py`: play a full terminal game against a trained checkpoint using MCTS.
- `config.yaml`: central config for paths, dataset handles/URLs, logging, and keys.
- `pyproject.toml`: Python project metadata and dependencies.
- `training-stages.md`: training roadmap and stage strategy notes.

### Source Code (`src/`)

- `src/model/models.py`
  - Model definitions: residual blocks, bottleneck variants, attention modules, `CheckersNetV1/V2/V3`.
- `src/model/trainer.py`
  - Training loop implementation, logging, early stopping, AMP, optional DDP wrapping, checkpoint lifecycle.
- `src/preprocess/process_datasets.py`
  - Dataset-specific normalization/parsing logic for all supported data sources.
- `src/preprocess/build_features.py`
  - Parquet cache materialization, distributed samplers, dataloader creation, train/test splits.
- `src/preprocess/datasets.py`
  - `ChessDataset` with chunk-level lazy loading and byte-to-tensor decoding.
- `src/common/tools.py`
  - Core helpers: logging, FEN and move encoding/decoding, SAN/UCI expansion, cp/mate handling, utility transforms.
- `src/common/utils.py`
  - Losses, metrics (policy top-k, value metrics), DDP setup/cleanup, model save utilities, writer helpers.
- `src/data/download.py`
  - Kaggle and direct-download data acquisition with retry/resume and `.zst` decompression.
- `src/evaluate/mcts.py`
  - MCTS node/search logic and network-guided expansion/backpropagation.

### Data and Artifacts

- `data/`: local datasets (raw and/or processed).
- `models/`: saved model checkpoints (`.pt`) and model notes.
- `results/`: serialized training metrics per run (`*_results.json`).
- `logs/`: training and utility logs.
- `temp_checkpoints/`: intermediate checkpoints during training.
- `notebooks/`: exploratory tests/analysis and model prediction notebooks.

## How to Run

### 1) Install dependencies

Using `uv` (recommended, matches project lockfile style):

```bash
uv sync
```

Or with `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2) Configure dataset credentials and paths

Edit `config.yaml`:

- set `kaggle_username`
- set `kaggle_api_key`
- verify data/model/result/log paths

### 3) Train

Example training command:

```bash
python train.py \
	--model-name checkers \
	--num-epochs 20 \
	--batch-size 512 \
	--learning-rate 1e-4 \
	--weight-decay 1e-4 \
	--train-policy \
	--train-value \
	--early-stopping-metric combined
```

To restrict datasets:

```bash
python train.py --model-name checkers --specified-datasets stockfish_position_evaluations,lichess_db_eval
```

### 4) Play against the model

```bash
python play_against_ai_terminal.py --model-name checkers4.2 --num-simulations 400
```

### 5) Run Elo benchmarks (cutechess-cli + Ordo)

This repo now includes a reproducible Elo workflow under `scripts/elo/`:

- `run_cutechess_match.sh`: runs engine-vs-engine matches and writes PGN + log.
- `run_ordo_rating.sh`: computes ratings from the match PGN with Ordo.
- `run_elo_pipeline.sh`: runs both steps in sequence.

Setup:

```bash
cp scripts/elo/elo.env.example scripts/elo/elo.env
```

Edit `scripts/elo/elo.env` and set at minimum:

- `ENGINE_A_CMD` to your Checkers command (for example `./checkers_engine --model-name checkers4.2 ...`)
- `ENGINE_B_CMD` to `stockfish` or your Stash binary
- `OPENINGS_FILE` to your `8moves_v3.pgn` absolute path

Run matches only:

```bash
scripts/elo/run_cutechess_match.sh
```

Run Ordo only (uses `RUN_TAG` from env or latest run under `results/elo/`):

```bash
scripts/elo/run_ordo_rating.sh
```

Run full pipeline:

```bash
scripts/elo/run_elo_pipeline.sh
```

Outputs are created under `results/elo/<RUN_TAG>/`:

- `matches.pgn` (or `PGN_NAME`)
- `cutechess.log`
- `ordo_ratings.txt`

Notes:

- The scripts are host-agnostic: set `CUTECHESS_BIN`/`ORDO_BIN` to absolute paths if binaries are not on `PATH`.
- Engine options are configured with `ENGINE_A_OPTIONS` and `ENGINE_B_OPTIONS` as semicolon-separated `Name=Value` pairs.
- You can run these scripts on your other Linux machine by copying this repo there and editing `scripts/elo/elo.env` for that host.

#### Rating Pool (Multi-Engine Gauntlet)

For more reliable Elo estimation, use a **rating pool** of engines at different levels:

```bash
cp scripts/elo/elo.env.preset-rating-pool scripts/elo/elo.env
```

Edit the pool definition (Stockfish skill levels, Stash, etc.):

```bash
scripts/elo/run_elo_gauntlet.sh
```

This will:

1. Run matches against each pooled opponent sequentially.
2. Combine all PGNs.
3. Compute composite Elo using Ordo.

**Why use a pool?**

- If your engine loses 95%+ against one super-strong engine, Elo is hard to estimate.
- Multiple opponents at different levels give you a bell curve of results (30-70% expected scores).
- Smaller error bars and better ranking reliability.

**Recommended pool (for ~1000-1200 total games, < 25 Elo error):**

- Stockfish skill 12 (medium): 400 games
- Stockfish skill 18 (strong): 300 games
- Stash (very strong): 200-300 games
- Stockfish skill 8 (weak): 200 games (optional, for variance)

**Stockfish skill level mapping** (0–20):

- Skill 0–5: ~1000 Elo (very weak)
- Skill 8–10: ~2100 Elo (club player)
- Skill 12–14: ~2600–2800 Elo (master)
- Skill 18–20: ~3400+ Elo (superhuman)

## Experiment Tracking

Training runs produce:

- checkpoint files in `models/`
- per-run metrics in `results/`
- logs in `logs/`
- optional TensorBoard logs when experiment args are provided

Metrics tracked include:

- learning rate schedule
- train/test loss
- value score
- policy top-1 accuracy
- policy top-3 accuracy

## Design Choices Worth Noting

- Policy representation moved to AlphaZero-style 73 planes, reducing output dimensionality compared with flat move-index encodings.
- Cached pre-encoded datasets significantly reduce repeated CPU preprocessing during long runs.
- The trainer is designed to degrade gracefully between DDP and non-DDP environments.
- Keyboard interrupt handling is implemented to avoid distributed deadlocks during training interruption.

## Known Gaps and Next High-Value Work

- Standardize cross-version metric comparability (especially across different policy spaces).
- Add automated benchmark suite against fixed engine/human test sets.
- Expand documentation for hyperparameter presets and recommended training curricula per hardware tier.
- Add stronger integration tests for dataset processing and cached encoding correctness.

## Why This Project Exists

Checkers started as a personal challenge and evolved into a full chess ML engineering project: model architecture iteration, large-scale dataset handling, robust training infrastructure, and search-based inference. The codebase now supports both experimentation and practical gameplay against trained checkpoints.

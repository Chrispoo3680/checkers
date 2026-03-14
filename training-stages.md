# Checkers 4.x - Training Roadmap (Updated to Current Code)

This document reflects the code currently in the repository after the recent architecture and pipeline updates.

## Current Implemented State

### Model architecture (implemented)

- Model: CheckersNetV3
- Trunk: 15 full-width ResidualBlocks (256 channels) with periodic SE blocks
- Policy head: AlphaZero-style conv head with 8x8x73 logits
- Value head: global average pooling + MLP
- Temperature: 1.0

This is no longer the old V3 bottleneck + giant FC policy head. The biggest architecture upgrade is already done.

### Policy representation (implemented)

- Policy output space is now AlphaZero-style 8x8x73 (4672 logits), not 20480.
- UCI move encoding/decoding is mapped into the 8x8x73 policy space.
- Policy targets in the dataset pipeline are generated in 8x8x73 shape.
- Loss and policy metrics flatten policy tensors internally, so both [B, 8, 8, 73] and [B, 4672] are handled.

### Training entrypoint (implemented)

- train.py now instantiates CheckersNetV3 with policy_planes=73.
- Early stopping is configurable by metric:
  - combined
  - policy
  - value

### Data pipeline status (implemented)

- lichess_db_eval JSONL is already supported in the preprocessing/dataloader pipeline.
- stockfish_position_evaluations CSV is already supported.
- magnus_carlsen_games CSV is already supported.
- lichess_puzzles parquet is already supported.

## Important operational note after policy-space migration

Because move indices changed from 20480 to 8x8x73, any old cached training artifacts that contain precomputed move indices from the previous encoding are incompatible.

If you have existing .training cache folders created before the migration, clear/rebuild them before training.

---

## Dataset Priority (unchanged)

| Dataset                        |            Positions | Policy Signal         | Value Signal | Quality       |
| ------------------------------ | -------------------: | --------------------- | ------------ | ------------- |
| lichess_db_eval                |         ~362,000,000 | multi-PV with cp/mate | cp/mate      | Excellent     |
| stockfish_position_evaluations |              468,235 | top-5 moves           | cp/mate      | Very high     |
| lichess_puzzles                |           ~5,750,000 | tactical forced moves | none         | High (policy) |
| magnus_carlsen_games           | ~1,500,000 positions | human expert moves    | W/D/L        | Medium        |

---

## Stage 1 - Foundation (lichess_db_eval + stockfish)

Goal: train both heads on engine-evaluated positions with broad coverage.

Data:

- lichess_db_eval (primary)
- stockfish_position_evaluations (supplementary)

Policy targets:

- Use deepest eval from lichess_db_eval per position
- Use all PV lines available at deepest depth
- Use cp-weighted softmax over PV moves:

  weight_i = softmax(-delta_cp_i / tau), where delta_cp_i = cp_best - cp_i

- Keep legal-move smoothing epsilon=0.02

Hyperparameters:
| Parameter | Value |
|---|---|
| Epochs | 30-50 |
| Batch size | 256 |
| Learning rate | 3e-4 |
| LR schedule | Cosine annealing to 1e-6 |
| Weight decay | 1e-4 |
| Value loss weight | 3.0 |
| Entropy weight | 0.01 |
| Gradient clipping | 1.0 |
| Checkpoint | none (from scratch) |
| Frozen blocks | none |
| Early stopping metric | combined |
| Early stopping patience | 5 |
| Early stopping delta | 0.001 |

Expected outcome:

- strong base policy/value features
- robust evaluation calibration

---

## Stage 2 - Tactical Sharpening (lichess_puzzles)

Goal: improve tactical move selection.

Data:

- lichess_puzzles expanded to position -> move pairs

Training mode:

- policy-only stage
- freeze value head

Hyperparameters:
| Parameter | Value |
|---|---|
| Epochs | 15-20 |
| Batch size | 256 |
| Learning rate | 5e-5 |
| LR schedule | Cosine annealing to 1e-6 |
| Weight decay | 1e-4 |
| Value loss weight | 0 (disabled) |
| Entropy weight | 0.01 |
| Gradient clipping | 1.0 |
| Checkpoint | Stage 1 best model |
| Freeze value head | yes |
| Early stopping metric | policy |
| Early stopping patience | 3 |
| Early stopping delta | 0.001 |

Expected outcome:

- improved tactical policy top-1/top-3

---

## Stage 3 - Human Style Integration (Magnus games)

Goal: add practical human-style decision preferences while preserving engine-grounded strength.

Data:

- magnus_carlsen_games

Training mode:

- both heads
- reduced value loss weight due to noisier W/D/L target

Hyperparameters:
| Parameter | Value |
|---|---|
| Epochs | 10-15 |
| Batch size | 256 |
| Learning rate | 2e-5 |
| LR schedule | Cosine annealing to 1e-6 |
| Weight decay | 1e-4 |
| Value loss weight | 1.0 |
| Entropy weight | 0.01 |
| Gradient clipping | 1.0 |
| Checkpoint | Stage 2 best model |
| Frozen blocks | 0-7 |
| Early stopping metric | policy |
| Early stopping patience | 3 |
| Early stopping delta | 0.001 |

Expected outcome:

- better practical move selection in near-equal positions

---

## Stage 4 (Optional) - Mixed Consolidation

Goal: reduce catastrophic forgetting after stagewise finetuning.

Data mix:

- 200K lichess_db_eval
- all stockfish_position_evaluations
- 200K lichess_puzzles
- all magnus positions

Hyperparameters:
| Parameter | Value |
|---|---|
| Epochs | 5-10 |
| Batch size | 256 |
| Learning rate | 1e-5 |
| LR schedule | Cosine annealing to 1e-6 |
| Weight decay | 1e-4 |
| Value loss weight | 3.0 |
| Checkpoint | Stage 3 best model |
| Frozen blocks | none |
| Early stopping metric | combined |
| Early stopping patience | 3 |
| Early stopping delta | 0.001 |

---

## Remaining High-Impact TODOs

1. Increase lichess_puzzles effective sample usage (currently still conservative in many runs).
2. Ensure cp-weighted softmax is consistently used for all multi-move engine-supervised sources (not only fixed weights).
3. Clear/rebuild old caches created with pre-migration (20480) move indexing.
4. Run a clean benchmark sweep comparing:
   - old 20480 policy checkpoints
   - new 8x8x73 policy checkpoints
     using identical evaluation protocol.

---

## Training Pipeline at a Glance

```
┌──────────────────────────────────────────────────────────┐
│  Stage 1: Foundation (lichess_db_eval + stockfish_evals) │
│  10–15M + 468K positions · both heads · LR 3e-4          │
│  30–50 epochs · from scratch · cp-weighted softmax       │
└─────────────────────┬────────────────────────────────────┘
                      │ checkpoint
┌─────────────────────▼────────────────────────────────────┐
│  Stage 2: Tactics (lichess_puzzles)                      │
│  1–2M puzzles · policy only · LR 5e-5                    │
│  15–20 epochs · value head frozen                        │
└─────────────────────┬────────────────────────────────────┘
                      │ checkpoint
┌─────────────────────▼────────────────────────────────────┐
│  Stage 3: Human Style (magnus_carlsen_games)             │
│  1.5M positions · both heads · LR 2e-5                   │
│  10–15 epochs · early blocks frozen                      │
└─────────────────────┬────────────────────────────────────┘
                      │ checkpoint (optional)
┌─────────────────────▼────────────────────────────────────┐
│  Stage 4: Consolidation (mixed dataset)                  │
│  ~2.4M positions · both heads · LR 1e-5                  │
│  5–10 epochs                                             │
└──────────────────────────────────────────────────────────┘
```

import math
import os
import random
import time
from pathlib import Path
from typing import Any, Callable

import chess
import numpy as np
import polars as pl
import torch
import torch.distributed as dist
from torch.utils.data import ConcatDataset, DataLoader, Sampler, Subset

from ..common import tools, utils
from .datasets import ChessDataset
from .process_datasets import (
    process_lichess_db_evals,
    process_lichess_puzzles,
    process_magnus_carlsen_games_csv,
    process_stockfish_evaluations_csv,
)

NUM_WORKERS: int = 0 if os.cpu_count() is None else os.cpu_count()  # type: ignore

_PARQUET_ROW_GROUP_SIZE = 10_000
_CACHE_BUILD_BATCH_SIZE = max(
    _PARQUET_ROW_GROUP_SIZE,
    int(os.getenv("CACHE_BUILD_BATCH_SIZE", "100000")),
)


def _materialize_compact_batch(batch_df: pl.DataFrame) -> pl.DataFrame:
    """Project a processed batch into training-ready encoded columns."""
    schema_names = set(batch_df.columns)
    if "fen" not in schema_names:
        raise ValueError("Processed dataset batch must include 'fen'")

    has_compact_policy = {
        "best_move_indices",
        "legal_move_indices",
        "normalized_evaluation",
    }.issubset(schema_names)

    if has_compact_policy:
        compact_batch = batch_df.select(
            pl.col("fen"),
            (
                pl.col("move_weights")
                if "move_weights" in schema_names
                else pl.lit(None)
            ).alias("move_weights"),
            pl.col("best_move_indices"),
            pl.col("legal_move_indices"),
            pl.col("normalized_evaluation"),
        )
    else:
        moves_expr = pl.col("moves") if "moves" in schema_names else pl.lit(None)
        evaluation_expr = (
            pl.col("evaluation") if "evaluation" in schema_names else pl.lit(None)
        )
        move_weights_expr = (
            pl.col("move_weights") if "move_weights" in schema_names else pl.lit(None)
        )

        compact_batch = batch_df.select(
            pl.col("fen"),
            move_weights_expr.alias("move_weights"),
            moves_expr.map_elements(
                _encode_best_move_indices, return_dtype=pl.List(pl.Int32)
            ).alias("best_move_indices"),
            pl.col("fen")
            .map_elements(_encode_legal_move_indices, return_dtype=pl.List(pl.Int32))
            .alias("legal_move_indices"),
            evaluation_expr.map_elements(
                _normalize_evaluation_scalar, return_dtype=pl.Float64
            ).alias("normalized_evaluation"),
        )

    return compact_batch.select(
        pl.col("fen")
        .map_elements(_encode_fen_to_bytes_f16, return_dtype=pl.Binary)
        .alias("encoded_board"),
        pl.struct(["best_move_indices", "legal_move_indices", "move_weights"])
        .map_elements(_encode_policy_target_to_bytes_f16, return_dtype=pl.Binary)
        .alias("encoded_policy_target"),
        pl.col("normalized_evaluation")
        .map_elements(_encode_value_target_to_bytes_f16, return_dtype=pl.Binary)
        .alias("encoded_value_target"),
    )


def _encode_best_move_indices(moves: str | None) -> list[int] | None:
    if moves is None or moves == "":
        return None

    return [tools.encode_UCI_to_int(move.strip()) for move in moves.split(",") if move]


def _encode_legal_move_indices(fen: str) -> list[int]:
    board = chess.Board(fen)
    return [tools.encode_UCI_to_int(move.uci()) for move in board.legal_moves]


def _normalize_evaluation_scalar(evaluation) -> float:
    if evaluation is None:
        return 0.0

    if isinstance(evaluation, str):
        if evaluation[0] == "M" and evaluation[1] == "-":
            evaluation = -1000
        elif evaluation[0] == "M":
            evaluation = 1000

        return float(torch.tanh(torch.tensor(float(evaluation) / 600)).item())

    value = float(evaluation)
    if not np.isfinite(value):
        return 0.0
    return float(np.tanh(np.clip(value, -1000.0, 1000.0) / 600.0))


def _encode_fen_to_bytes_f16(fen: str) -> bytes:
    """Encode a FEN string into an 18x8x8 float16 board tensor serialized as raw bytes."""
    return tools.encode_fen_pos(fen).to(dtype=torch.float16).numpy().tobytes()


def _encode_policy_target_to_bytes_f16(policy_row: dict[str, Any]) -> bytes:
    best_move_indices = policy_row.get("best_move_indices")
    legal_move_indices = policy_row.get("legal_move_indices")
    move_weights = policy_row.get("move_weights")

    if not best_move_indices:
        policy = torch.zeros((8, 8, tools.POLICY_PLANES), dtype=torch.float16)
        return policy.numpy().tobytes()

    if legal_move_indices is None:
        raise ValueError(
            "legal_move_indices is required to build cached policy targets"
        )

    policy = utils.create_policy_distribution_with_smoothing(
        top_move_indices=list(best_move_indices),
        legal_move_indices=list(legal_move_indices),
        move_weights=(list(move_weights) if move_weights is not None else None),
    ).view(8, 8, tools.POLICY_PLANES)

    return policy.to(dtype=torch.float16).numpy().tobytes()


def _encode_value_target_to_bytes_f16(normalized_evaluation: float | None) -> bytes:
    value = 0.0 if normalized_evaluation is None else float(normalized_evaluation)
    if not np.isfinite(value):
        value = 0.0
    value = float(np.clip(value, -1.0, 1.0))
    return np.asarray([value], dtype=np.float16).tobytes()


class BlockDistributedSampler(Sampler[int]):
    def __init__(
        self,
        dataset,
        block_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ):
        if dist.is_available() and dist.is_initialized():
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.num_replicas = 1
            self.rank = 0

        self.dataset = dataset
        self.block_size = block_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)

    def __iter__(self):
        generator = random.Random(self.seed + self.epoch)
        block_starts = list(range(0, len(self.dataset), self.block_size))

        if self.shuffle:
            generator.shuffle(block_starts)

        if not self.drop_last and len(block_starts) % self.num_replicas != 0:
            pad = self.num_replicas - (len(block_starts) % self.num_replicas)
            block_starts.extend(block_starts[:pad])
        elif self.drop_last:
            usable = len(block_starts) - (len(block_starts) % self.num_replicas)
            block_starts = block_starts[:usable]

        local_block_starts = block_starts[self.rank :: self.num_replicas]
        indices: list[int] = []
        for block_start in local_block_starts:
            block_indices = list(
                range(
                    block_start, min(block_start + self.block_size, len(self.dataset))
                )
            )
            if self.shuffle:
                generator.shuffle(block_indices)
            indices.extend(block_indices)

        if len(indices) < self.num_samples and indices:
            indices.extend(indices[: self.num_samples - len(indices)])

        return iter(indices[: self.num_samples])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int):
        self.epoch = epoch


def _get_cached_parquet(
    source_path: Path, build_lazy_df: Callable[[], pl.LazyFrame]
) -> pl.LazyFrame:
    """Convert a processed LazyFrame to a cached Parquet file for fast random access.

    Uses the streaming sink when possible; falls back to collect + write for lazy
    frames that contain Python UDFs (e.g. map_elements) unsupported by the
    streaming engine. The cache file sits next to the source file and is reused
    on all subsequent runs.
    """
    cache_path = source_path.parent / (source_path.stem + ".training_az73")
    ready_file = cache_path / ".complete"
    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0

    if rank == 0 and not ready_file.exists():
        print(
            f"  Creating Parquet cache at {cache_path} "
            f"(one-time conversion, may take a few minutes)..."
        )
        cache_path.mkdir(parents=True, exist_ok=True)
        for part_file in cache_path.glob("*.parquet"):
            part_file.unlink()
        if ready_file.exists():
            ready_file.unlink()

        lazy_df = build_lazy_df()
        total_rows = lazy_df.select(pl.len()).collect().item()
        started_at = time.perf_counter()
        batches_total = math.ceil(total_rows / _CACHE_BUILD_BATCH_SIZE)
        for batch_num, offset in enumerate(
            range(0, total_rows, _CACHE_BUILD_BATCH_SIZE), start=1
        ):
            batch_df = lazy_df.slice(offset, _CACHE_BUILD_BATCH_SIZE).collect()
            cached_batch_df = _materialize_compact_batch(batch_df)
            cached_batch_df.write_parquet(
                cache_path / f"part_{offset:09d}.parquet",
                row_group_size=_PARQUET_ROW_GROUP_SIZE,
            )

            if batch_num % 1 == 0 or batch_num == batches_total:
                elapsed = max(time.perf_counter() - started_at, 1e-6)
                done_rows = min(offset + _CACHE_BUILD_BATCH_SIZE, total_rows)
                rows_per_sec = int(done_rows / elapsed)
                print(
                    f"  Cache progress: {batch_num}/{batches_total} batches "
                    f"({done_rows:,}/{total_rows:,} rows, {rows_per_sec:,} rows/s)"
                )

        ready_file.touch()
        print("  Parquet cache created.")

    if is_dist:
        if torch.cuda.is_available() and dist.get_backend() == "nccl":
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()

    if not ready_file.exists():
        raise RuntimeError(f"Cache build did not complete for {cache_path}")

    return pl.scan_parquet(cache_path / "*.parquet")


def _split_dataset(dataset: ChessDataset) -> tuple[Subset, Subset]:
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset = Subset(dataset, range(0, train_size))
    test_dataset = Subset(dataset, range(train_size, train_size + test_size))
    return train_dataset, test_dataset


def create_dataloaders(
    data_dir_paths: list[Path],
    batch_size: int,
    num_workers: int = NUM_WORKERS,
    cache_decode_dtype: torch.dtype = torch.float32,
):

    # Make data folders into dataset
    train_datasets: list[Subset] = []
    test_datasets: list[Subset] = []
    for path in data_dir_paths:
        if any("training " in part for part in path.parts):
            continue  # Skip already processed parquet files

        print(f"Processing dataset from {path.parent.stem.lower()}...")

        if path.suffix == ".jsonl" and "lichess_db_eval" in path.parent.stem.lower():
            lazy_df = _get_cached_parquet(
                path, lambda p=path: process_lichess_db_evals(p)
            )
        elif (
            path.suffix == ".parquet" and "lichess_puzzles" in path.parent.stem.lower()
        ):
            lazy_df = _get_cached_parquet(
                path, lambda p=path: process_lichess_puzzles(p)
            )

        elif (
            path.suffix == ".csv"
            and "stockfish_position_evaluations" in path.parent.stem.lower()
        ):
            lazy_df = _get_cached_parquet(
                path, lambda p=path: process_stockfish_evaluations_csv(p)
            )

        elif (
            path.suffix == ".csv" and "magnus_carlsen_games" in path.parent.stem.lower()
        ):
            lazy_df = _get_cached_parquet(
                path, lambda p=path: process_magnus_carlsen_games_csv(p)
            )
        else:
            continue

        dataset = ChessDataset(
            lazy_df=lazy_df,
            chunk_size=_PARQUET_ROW_GROUP_SIZE,
            decode_dtype=cache_decode_dtype,
        )
        print(
            f"Loading dataset from {path.parent.stem.lower()} with {len(dataset)} samples."
        )
        train_dataset, test_dataset = _split_dataset(dataset)
        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)

    train_dataset = ConcatDataset(train_datasets)
    test_dataset = ConcatDataset(test_datasets)

    # Make dataset into dataloader
    train_dataloader_kwargs = {
        "dataset": train_dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "sampler": BlockDistributedSampler(
            train_dataset,
            block_size=_PARQUET_ROW_GROUP_SIZE,
            shuffle=True,
        ),
    }
    test_dataloader_kwargs = {
        "dataset": test_dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "sampler": BlockDistributedSampler(
            test_dataset,
            block_size=_PARQUET_ROW_GROUP_SIZE,
            shuffle=False,
        ),
    }

    if num_workers > 0:
        train_dataloader_kwargs["persistent_workers"] = True
        train_dataloader_kwargs["prefetch_factor"] = 4
        train_dataloader_kwargs["multiprocessing_context"] = "spawn"
        train_dataloader_kwargs["timeout"] = 120
        test_dataloader_kwargs["persistent_workers"] = True
        test_dataloader_kwargs["prefetch_factor"] = 4
        test_dataloader_kwargs["multiprocessing_context"] = "spawn"
        test_dataloader_kwargs["timeout"] = 120

    train_dataloader = DataLoader(**train_dataloader_kwargs)
    test_dataloader = DataLoader(**test_dataloader_kwargs)

    return (train_dataloader, test_dataloader), (train_dataset, test_dataset)

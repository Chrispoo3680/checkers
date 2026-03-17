from collections import OrderedDict
from typing import Sequence, cast

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from ..common import tools


class ChessDataset(Dataset):
    def __init__(
        self,
        lazy_df: pl.LazyFrame,
        chunk_size: int = 10_000,
        max_cached_chunks: int = 8,
        decode_dtype: torch.dtype = torch.float32,
    ):
        schema_names = lazy_df.collect_schema().names()
        required_columns = {
            "encoded_board",
            "encoded_policy_target",
            "encoded_value_target",
        }
        missing_columns = required_columns.difference(schema_names)
        if missing_columns:
            raise ValueError(
                f"Missing required encoded cache columns: {sorted(missing_columns)}"
            )

        self.lazy_df = lazy_df
        self.chunk_size = chunk_size
        self.max_cached_chunks = max_cached_chunks
        self.decode_dtype = decode_dtype
        self._length = None
        self._chunk_cache: OrderedDict[int, pl.DataFrame] = OrderedDict()

    def _get_length(self):
        if self._length is None:
            self._length = self.lazy_df.select(pl.len()).collect().item()
        return self._length

    def _get_chunk(self, chunk_idx: int) -> pl.DataFrame:
        chunk_df = self._chunk_cache.get(chunk_idx)
        if chunk_df is None:
            chunk_df = self.lazy_df.slice(
                chunk_idx * self.chunk_size, self.chunk_size
            ).collect()
            self._chunk_cache[chunk_idx] = chunk_df
            if len(self._chunk_cache) > self.max_cached_chunks:
                self._chunk_cache.popitem(last=False)
        else:
            self._chunk_cache.move_to_end(chunk_idx)

        return chunk_df

    def _decode_encoded_row(
        self,
        encoded_board_bytes: bytes | None,
        encoded_policy_bytes: bytes | None,
        encoded_value_bytes: bytes | None,
    ):
        if encoded_board_bytes is None:
            raise ValueError("encoded_board cannot be null in cached datasets")

        if self.decode_dtype == torch.float16:
            decoded_position = torch.from_numpy(
                np.frombuffer(encoded_board_bytes, dtype=np.float16)
                .reshape(18, 8, 8)
                .copy()
            )

            if encoded_policy_bytes is None:
                encoded_best_moves = torch.zeros(
                    (8, 8, tools.POLICY_PLANES), dtype=torch.float16
                )
            else:
                encoded_best_moves = torch.from_numpy(
                    np.frombuffer(encoded_policy_bytes, dtype=np.float16)
                    .reshape(8, 8, tools.POLICY_PLANES)
                    .copy()
                )

            if encoded_value_bytes is None:
                normalized_evaluation = torch.zeros(1, dtype=torch.float16)
            else:
                normalized_evaluation = torch.from_numpy(
                    np.frombuffer(encoded_value_bytes, dtype=np.float16).copy()
                )
        else:
            decoded_position = torch.from_numpy(
                np.frombuffer(encoded_board_bytes, dtype=np.float16)
                .astype(np.float32)
                .reshape(18, 8, 8)
                .copy()
            )

            if encoded_policy_bytes is None:
                encoded_best_moves = torch.zeros(
                    (8, 8, tools.POLICY_PLANES), dtype=torch.float32
                )
            else:
                encoded_best_moves = torch.from_numpy(
                    np.frombuffer(encoded_policy_bytes, dtype=np.float16)
                    .astype(np.float32)
                    .reshape(8, 8, tools.POLICY_PLANES)
                    .copy()
                )

            if encoded_value_bytes is None:
                normalized_evaluation = torch.zeros(1, dtype=torch.float32)
            else:
                normalized_evaluation = torch.from_numpy(
                    np.frombuffer(encoded_value_bytes, dtype=np.float16)
                    .astype(np.float32)
                    .copy()
                )

        normalized_evaluation = torch.nan_to_num(
            normalized_evaluation, nan=0.0, posinf=1.0, neginf=-1.0
        )
        normalized_evaluation = torch.clamp(normalized_evaluation, -1.0, 1.0)

        return decoded_position, encoded_best_moves, normalized_evaluation

    def __getitem__(self, index):
        chunk_idx = index // self.chunk_size
        row_idx = index % self.chunk_size
        row = self._get_chunk(chunk_idx).row(row_idx)
        return self._decode_encoded_row(
            cast(bytes | None, row[0]),
            cast(bytes | None, row[1]),
            cast(bytes | None, row[2]),
        )

    def __getitems__(self, indices: Sequence[int]):
        samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None] = [
            None
        ] * len(indices)
        chunk_requests: OrderedDict[int, list[tuple[int, int]]] = OrderedDict()

        for sample_pos, index in enumerate(indices):
            chunk_idx = index // self.chunk_size
            row_idx = index % self.chunk_size
            chunk_requests.setdefault(chunk_idx, []).append((sample_pos, row_idx))

        for chunk_idx, requests in chunk_requests.items():
            chunk_df = self._get_chunk(chunk_idx)
            for sample_pos, row_idx in requests:
                row = chunk_df.row(row_idx)
                samples[sample_pos] = self._decode_encoded_row(
                    cast(bytes | None, row[0]),
                    cast(bytes | None, row[1]),
                    cast(bytes | None, row[2]),
                )

        return cast(list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], samples)

    def __len__(self):
        return self._get_length()

import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from ..common import tools
from .datasets import ChessDataset

NUM_WORKERS: int = 0 if os.cpu_count() is None else os.cpu_count()  # type: ignore


def create_dataloaders(
    data_dir_paths: list[Path],
    batch_size: int,
    num_workers: int = NUM_WORKERS,
):

    # Make data folders into dataset
    independent_datasets: list[ChessDataset] = []
    for path in data_dir_paths:
        filtered_df = pd.DataFrame()

        if path.suffix == ".parquet" and "lichess_puzzles" in path.parent.stem.lower():
            puzzles_df = pd.read_parquet(path, columns=["FEN", "Moves"])
            puzzles_df = puzzles_df.sample(n=50000, random_state=42).reset_index(
                drop=True
            )
            expanded_df = tools.expand_game_positions(puzzles_df)
            expanded_df["evaluation"] = None
            filtered_df = expanded_df.sample(n=50000, random_state=42)

        elif path.suffix == ".csv" and "stockfish" in path.parent.stem.lower():
            positions_df = pd.read_csv(path)
            filtered_df = positions_df[["fen", "evaluation"]]
            filtered_df["moves"] = (
                (
                    positions_df["move_1"].fillna("").astype(str)
                    + ","
                    + positions_df["move_2"].fillna("").astype(str)
                    + ","
                    + positions_df["move_3"].fillna("").astype(str)
                    + ","
                    + positions_df["move_4"].fillna("").astype(str)
                    + ","
                    + positions_df["move_5"].fillna("").astype(str)
                )
                .str.replace(r",+", ",", regex=True)
                .str.strip(",")
            )

        elif path.suffix == ".csv" and "magnus_carlsen" in path.parent.stem.lower():
            games_df = (
                pd.read_csv(
                    path,
                    usecols=["moves", "result_raw"],
                ).dropna()
                # .sample(n=5000, random_state=42)
                # .reset_index(drop=True)
            )
            # Keep result_raw as white-absolute perspective (+1 = white wins,
            # -1 = black wins). expand_game_positions_san alternates the sign
            # at each half-move using (-1)^i where i=0 is always white's first
            # move, so this correctly produces current-player-perspective labels
            # for every position. Flipping the sign based on which side Magnus
            # played inverts ALL labels in black-sided games (bug).
            games_df["result_raw"] = games_df["result_raw"].map(
                {"1-0": 1, "0-1": -1, "0.5-0.5": 0}
            )
            filtered_df = tools.expand_game_positions_san(
                games_df, moves_col="moves", eval_col="result_raw"
            )

        print(
            f"Loading dataset from {path.parent.stem.lower()} with {len(filtered_df)} samples."
        )
        independent_datasets.append(ChessDataset(data_df=filtered_df))

    full_dataset = ConcatDataset(independent_datasets)

    # Split into training and testing data (80% training, 20% testing)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_dataset, test_dataset = random_split(
        full_dataset, [train_size, test_size], torch.manual_seed(0)
    )

    # Make dataset into dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        sampler=DistributedSampler(train_dataset),
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        sampler=DistributedSampler(test_dataset),
    )

    return (train_dataloader, test_dataloader), (train_dataset, test_dataset)

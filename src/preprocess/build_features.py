import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

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
        # puzzles_df = pd.read_parquet(path, columns=["FEN", "Moves"])
        # expanded_df = tools.expand_game_positions(puzzles_df[:10000])
        # expanded_df["evaluation"] = None

        # games_df = pd.read_csv(
        #     path,
        #     usecols=["moves"],
        # ).dropna()
        # expanded_df = tools.expand_game_positions_san(
        #     games_df[5000:10000], moves_col="moves"
        # )
        # expanded_df["evaluation"] = None

        games_df = pd.read_csv(
            path,
            usecols=["Position", "Best Move", "Evaluation"],
        ).dropna()
        games_df.rename(
            columns={
                "Position": "fen",
                "Best Move": "move",
                "Evaluation": "evaluation",
            },
            inplace=True,
        )

        independent_datasets.append(ChessDataset(data_df=games_df))

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

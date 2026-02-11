import os
from pathlib import Path

from torch.utils.data import DataLoader

from .datasets import ChessDataset

NUM_WORKERS: int = 0 if os.cpu_count() is None else os.cpu_count()  # type: ignore


def create_dataloaders(
    train_data_path: Path,
    test_data_path: Path,
    batch_size: int,
    num_workers: int = NUM_WORKERS,
):

    # Make data folders into dataset
    train_dataset = ChessDataset(
        train_data_path,
    )
    test_dataset = ChessDataset(
        test_data_path,
    )

    # Make dataset into dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    return (train_dataloader, test_dataloader), (train_dataset, test_dataset)

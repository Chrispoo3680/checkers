"""
Contains various utility functions for PyTorch model training and saving.
"""

import torch
from torch.utils.tensorboard.writer import SummaryWriter
import torch.distributed as dist

import os
from pathlib import Path
from . import tools

from typing import List, Union, Dict, Any, Optional


try:
    logging_file_path = os.environ["LOGGING_FILE_PATH"]
except KeyError:
    logging_file_path = None

logger = tools.create_logger(log_path=logging_file_path, logger_name=__name__)


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    torch.cuda.set_device(rank)

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def ddp_cleanup():
    dist.destroy_process_group()


def save_model(
    model: Union[torch.nn.Module, Dict[str, Any]],
    target_dir_path: Path,
    model_name: str,
):

    # Create target directory
    os.makedirs(target_dir_path, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt")

    model_save_path: Path = target_dir_path / model_name

    if isinstance(model, torch.nn.Module):
        model_state_dict = model.state_dict()
    else:
        model_state_dict = model

    # Save the model state_dict()
    logger.info(f"  Saving model to: {model_save_path}")
    torch.save(obj=model_state_dict, f=model_save_path)


def model_save_version(save_dir_path: Path, save_name: str) -> str:

    files_in_dir: List[str] = os.listdir(save_dir_path)
    version = str(sum([1 for file in files_in_dir if save_name in file]))

    save_name_version: str = f"{save_name}V{version}"

    return save_name_version


def create_writer(
    root_dir: Path,
    experiment_name: str,
    model_name: str,
    var: str,
) -> SummaryWriter:
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    Args:
        root_dir (Path): Root dir of the repository.
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        var (str): The varying factor to change when experimenting on what gives the best results. Also called the 'independent variable'.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.
    """

    writer_log_dir: str = os.path.join(
        root_dir, "runs/classification", experiment_name, model_name, var
    )

    logger.info(f"Created SummaryWriter, saving to:  {writer_log_dir}...")

    return SummaryWriter(log_dir=writer_log_dir)


class EarlyStopping:
    def __init__(self, patience: int = 5, delta: float = 0) -> None:
        """
        Args:
            patience (int): How many epochs to wait after the last improvement.
            min_delta (float): Minimum change to qualify as an improvement.
        """
        self.patience: int = patience
        self.delta: float = delta
        self.best_score: float = float("inf")
        self.best_score_epoch: int = 0
        self.early_stop: bool = False
        self.counter: int = 0
        self.best_model_state: Dict[str, Any] = {}

    def __call__(self, val_loss, model, epoch) -> None:
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_score_epoch = epoch
            self.best_model_state = model.state_dict()
        elif score > self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score_epoch = epoch
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model) -> None:
        model.load_state_dict(self.best_model_state)

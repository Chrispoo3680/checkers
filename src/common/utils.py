"""
Contains various utility functions for PyTorch model training and saving.
"""

import os
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter

from . import tools

try:
    logging_file_path = os.environ["LOGGING_FILE_PATH"]
except KeyError:
    logging_file_path = None

logger = tools.create_logger(log_path=logging_file_path, logger_name=__name__)


def _flatten_policy_tensor(policy_tensor: torch.Tensor) -> torch.Tensor:
    if policy_tensor.dim() > 2:
        return policy_tensor.view(policy_tensor.size(0), -1)
    return policy_tensor


def chess_loss(
    p_logits,
    v_pred,
    pi_target=None,
    v_target=None,
    value_weight=1.0,
    entropy_weight=0.01,
):

    policy_loss = None
    value_loss = None
    entropy = None
    total_loss = torch.zeros(1, device=p_logits.device, dtype=p_logits.dtype)

    if pi_target is not None:
        p_logits = _flatten_policy_tensor(p_logits)
        pi_target = _flatten_policy_tensor(pi_target)
        log_probs = F.log_softmax(p_logits, dim=1)
        policy_loss = -(pi_target * log_probs).sum(dim=1).mean()
        entropy = -(log_probs.exp() * log_probs).sum(dim=1).mean()
        total_loss = total_loss + policy_loss - entropy_weight * entropy

    if v_target is not None:
        v_target = torch.nan_to_num(v_target, nan=0.0, posinf=1.0, neginf=-1.0)
        v_target = torch.clamp(v_target, -1.0, 1.0)
        v_pred = torch.nan_to_num(v_pred, nan=0.0, posinf=1.0, neginf=-1.0)
        value_loss = F.mse_loss(v_pred, v_target)
        total_loss = total_loss + value_weight * value_loss

    return total_loss, policy_loss, value_loss, entropy


def value_correlation(v_pred, v_target):

    v_pred = v_pred.view(-1).float()
    v_target = v_target.view(-1).float()

    v_pred = v_pred - v_pred.mean()
    v_target = v_target - v_target.mean()

    numerator = torch.sum(v_pred * v_target)
    denominator = torch.sqrt(torch.sum(v_pred**2) * torch.sum(v_target**2))
    denominator = torch.clamp(denominator, min=1e-8)

    corr = numerator / denominator

    return corr.item()


def value_sign_accuracy(v_pred, v_target):

    pred_sign = torch.sign(v_pred)
    target_sign = torch.sign(v_target)

    correct = (pred_sign == target_sign).float().mean()

    return correct.item()


def validation_score(correlation, sign_accuracy):
    return 0.7 * correlation + 0.3 * sign_accuracy


def policy_top1_accuracy(policy_logits, pi_target, legal_mask=None):
    """Returns (correct_count, total_count) for top-1 policy accuracy.
    Optionally masks illegal moves before taking the argmax of logits."""
    policy_logits = _flatten_policy_tensor(policy_logits)
    pi_target = _flatten_policy_tensor(pi_target)
    if legal_mask is not None:
        legal_mask = _flatten_policy_tensor(legal_mask)
        policy_logits = policy_logits.masked_fill(legal_mask == 0, float("-inf"))
    pred_top1 = torch.argmax(policy_logits, dim=1)
    target_top1 = torch.argmax(pi_target, dim=1)
    correct = (pred_top1 == target_top1).sum().item()
    total = pi_target.size(0)
    return correct, total


def policy_top3_accuracy(policy_logits, pi_target, legal_mask=None):
    """Returns (correct_count, total_count) for top-3 policy accuracy.
    A prediction is correct if the model's top-1 move is among the top-3
    moves in pi_target with non-zero probability. Handles positions with
    fewer than 3 legal moves correctly."""
    policy_logits = _flatten_policy_tensor(policy_logits)
    pi_target = _flatten_policy_tensor(pi_target)
    if legal_mask is not None:
        legal_mask = _flatten_policy_tensor(legal_mask)
        policy_logits = policy_logits.masked_fill(legal_mask == 0, float("-inf"))
    pred_top1 = torch.argmax(policy_logits, dim=1).unsqueeze(1)  # (B, 1)

    k = min(3, pi_target.size(1))
    top3 = torch.topk(pi_target, k=k, dim=1)
    target_top3_indices = top3.indices  # (B, k)
    target_top3_probs = top3.values  # (B, k)

    match = (pred_top1 == target_top3_indices) & (target_top3_probs > 0)
    correct = match.any(dim=1).sum().item()
    total = pi_target.size(0)
    return correct, total


def create_policy_distribution_with_smoothing(
    top_move_indices: list[int],
    legal_move_indices: list[int],
    move_weights: Optional[list[float]] = None,
    policy_size: int = tools.POLICY_SIZE,
    epsilon: float = 0.02,
):
    """
    Creates a policy distribution with smoothing.

    Args:
        top_move_indices: indices of top moves from Stockfish
        legal_move_indices: indices of all legal moves
        move_weights: weights for the top moves
        epsilon: probability mass for non-top moves

    Returns:
        A tensor of shape (policy_size,) representing the policy distribution.
    """

    if move_weights is None:
        move_weights = [0.5, 0.2, 0.15, 0.1, 0.05]

    policy = torch.zeros(policy_size, dtype=torch.float32)

    k = len(top_move_indices)
    base_weights = torch.tensor(move_weights[:k])
    base_weights = base_weights / base_weights.sum()

    top_indices = torch.tensor(top_move_indices, dtype=torch.long)
    policy[top_indices] = (1 - epsilon) * base_weights

    legal_indices = torch.tensor(legal_move_indices, dtype=torch.long)
    other_indices = legal_indices[~torch.isin(legal_indices, top_indices)]

    if other_indices.numel() > 0:
        policy[other_indices] = epsilon / other_indices.numel()

    policy = policy / policy.sum()

    return policy


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    torch.cuda.set_device(rank)

    try:
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            device_id=torch.device("cuda", rank),
        )
    except TypeError:
        # Backward compatibility with older torch versions without device_id.
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def ddp_cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def save_model(
    model: Union[torch.nn.Module, dict[str, Any]],
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

    files_in_dir: list[str] = os.listdir(save_dir_path)
    version = str(sum([1 for file in files_in_dir if save_name in file]))

    save_name_version: str = f"{save_name}{version}"

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
        self.best_score: float = float("-inf")
        self.best_score_epoch: int = 0
        self.early_stop: bool = False
        self.counter: int = 0
        self.best_model_state: dict[str, Any] = {}
        self.last_3_scores: list[float] = []

    def __call__(self, val_loss, model, epoch) -> None:
        score = val_loss

        self.last_3_scores.append(score)
        if len(self.last_3_scores) > 3:
            self.last_3_scores.pop(0)

        score_ma = float(np.mean(self.last_3_scores))

        if score_ma > self.best_score + self.delta:
            self.best_score = score_ma
            self.best_score_epoch = epoch
            self.best_model_state = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def load_best_model(self, model) -> None:
        model.load_state_dict(self.best_model_state)

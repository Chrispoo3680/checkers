import os
import signal
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from ..common import tools, utils

try:
    logging_file_path = os.environ["LOGGING_FILE_PATH"]
except KeyError:
    logging_file_path = None

logger = tools.create_logger(log_path=logging_file_path, logger_name=__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer,
        loss_fn: Callable,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        device: torch.device,
        rank: int,
        scaler: GradScaler,
        early_stopping: utils.EarlyStopping,
        train_policy: bool = True,
        train_value: bool = True,
        lr_scheduler: Optional[Union[MultiStepLR, StepLR, CosineAnnealingLR]] = None,
        temp_checkpoint_file_path: Optional[Path] = None,
        writer: Optional[SummaryWriter] = None,
    ):

        self.device = device
        self.rank = rank
        self.model = DDP(
            model.to(rank), device_ids=[rank], find_unused_parameters=False
        )
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_policy = train_policy
        self.train_value = train_value
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.scaler = scaler
        self.skip_lr_sched = False
        self.lr_scheduler = lr_scheduler
        self.early_stopping = early_stopping
        self.temp_checkpoint_file_path = temp_checkpoint_file_path
        self.writer = writer
        self._interrupted = False

    def _handle_sigint(self, signum, frame):
        self._interrupted = True

    def train_step(self, epoch: int):

        self.train_dataloader.sampler.set_epoch(epoch)  # type: ignore

        self.model.train()

        train_loss = 0

        for batch, (X, pi_target, value_target) in enumerate(
            tqdm(
                self.train_dataloader,
                position=1,
                leave=False,
                desc="Iterating through training batches",
                disable=self.rank != 0,
            )
        ):
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                X, pi_target, value_target = (
                    X.to(self.rank),
                    pi_target.to(self.rank),
                    value_target.to(self.rank),
                )

                policy_logist, value_pred = self.model(X)

                loss, policy_loss, value_loss, entropy = self.loss_fn(
                    policy_logist,
                    value_pred,
                    pi_target if self.train_policy else None,
                    value_target if self.train_value else None,
                    value_weight=3.0,
                )

                train_loss += loss.item()

            self.optimizer.zero_grad(set_to_none=True)

            self.scaler.scale(loss).backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.scaler.step(self.optimizer)

            scale = self.scaler.get_scale()
            self.scaler.update()
            self.skip_lr_sched = scale != self.scaler.get_scale()

        return train_loss

    def test_step(self, epoch: int):

        self.test_dataloader.sampler.set_epoch(epoch)  # type: ignore

        self.model.eval()

        test_loss = 0
        value_preds_list = []
        value_targets_list = []
        policy_top1_correct = 0
        policy_top1_total = 0
        policy_top3_correct = 0
        policy_top3_total = 0

        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            with torch.inference_mode():
                for batch, (X, pi_target, value_target) in enumerate(
                    tqdm(
                        self.test_dataloader,
                        position=1,
                        leave=False,
                        desc="Iterating through testing batches",
                        disable=self.rank != 0,
                    )
                ):
                    X, pi_target, value_target = (
                        X.to(self.rank),
                        pi_target.to(self.rank),
                        value_target.to(self.rank),
                    )

                    policy_logist, value_pred = self.model(X)

                    if self.train_value:
                        value_preds_list.append(value_pred.cpu())
                        value_targets_list.append(value_target.cpu())

                    if self.train_policy:
                        legal_mask = (pi_target > 0).float()
                        batch_correct, batch_total = utils.policy_top1_accuracy(
                            policy_logist.cpu(), pi_target.cpu(), legal_mask.cpu()
                        )
                        policy_top1_correct += batch_correct
                        policy_top1_total += batch_total
                        batch_top3_correct, batch_top3_total = (
                            utils.policy_top3_accuracy(
                                policy_logist.cpu(), pi_target.cpu(), legal_mask.cpu()
                            )
                        )
                        policy_top3_correct += batch_top3_correct
                        policy_top3_total += batch_top3_total

                    loss, policy_loss, value_loss, entropy = self.loss_fn(
                        policy_logist,
                        value_pred,
                        pi_target if self.train_policy else None,
                        value_target if self.train_value else None,
                        value_weight=3.0,
                    )

                    test_loss += loss.item()

        if self.train_value:
            all_value_preds = torch.cat(value_preds_list)
            all_value_targets = torch.cat(value_targets_list)
            corr = utils.value_correlation(all_value_preds, all_value_targets)
            sign_acc = utils.value_sign_accuracy(all_value_preds, all_value_targets)
            value_score = utils.validation_score(corr, sign_acc)
        else:
            value_score = 0.0

        if self.train_policy:
            policy_top1_acc = (
                policy_top1_correct / policy_top1_total
                if policy_top1_total > 0
                else 0.0
            )
            policy_top3_acc = (
                policy_top3_correct / policy_top3_total
                if policy_top3_total > 0
                else 0.0
            )
        else:
            policy_top1_acc = 0.0
            policy_top3_acc = 0.0

        return test_loss, value_score, policy_top1_acc, policy_top3_acc

    def train(self, epochs: int) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:

        results: Dict[str, List[float]] = {
            "learning_rate": [],
            "train_loss": [],
            "test_loss": [],
            "value_score": [],
            "policy_top1_accuracy": [],
            "policy_top3_accuracy": [],
        }

        # Suppress default KeyboardInterrupt on all ranks so a Ctrl+C cannot
        # fire inside a torch.distributed call and cause a deadlock.
        # The flag is checked at epoch boundaries and broadcast from rank 0.
        original_sigint_handler = signal.signal(signal.SIGINT, self._handle_sigint)

        try:
            for epoch in tqdm(
                range(epochs),
                position=0,
                desc="Iterating through epochs",
                disable=self.rank != 0,
            ):
                train_loss = self.train_step(epoch)
                test_loss, value_score, policy_top1_acc, policy_top3_acc = (
                    self.test_step(epoch)
                )

                results["learning_rate"].append(self.optimizer.param_groups[0]["lr"])
                results["train_loss"].append(train_loss)
                results["test_loss"].append(test_loss)
                results["value_score"].append(value_score)
                results["policy_top1_accuracy"].append(policy_top1_acc)
                results["policy_top3_accuracy"].append(policy_top3_acc)

                if self.rank == 0:
                    self.early_stopping(value_score, self.model.module, epoch + 1)

                    early_stop_flag = int(
                        self.early_stopping.early_stop or self._interrupted
                    )

                    # Log and save epoch loss and accuracy results
                    logger.info(
                        f"GPU ID: {self.rank} | "
                        f"epoch: {epoch+1:>3} | "
                        f"train loss: {train_loss:>10.4f} | "
                        f"test loss: {test_loss:>10.4f} | "
                        f"value score: {value_score:.4f} | "
                        f"pi top1 acc: {policy_top1_acc:.4f} | "
                        f"pi top3 acc: {policy_top3_acc:.4f} | "
                        f"lr: {self.optimizer.param_groups[0]['lr']:.2e} | "
                        f"stopping counter: {self.early_stopping.counter} / {self.early_stopping.patience}"
                    )

                    # See if there's a writer, if so, log to it
                    if self.writer:
                        self.writer.add_scalar(
                            tag="Learning rate",
                            scalar_value=self.optimizer.param_groups[0]["lr"],
                            global_step=epoch,
                        )
                        self.writer.add_scalars(
                            main_tag="Loss",
                            tag_scalar_dict={
                                "train_loss": train_loss,
                                "test_loss": test_loss,
                            },
                            global_step=epoch,
                        )
                        self.writer.add_scalar(
                            tag="Value score",
                            scalar_value=value_score,
                            global_step=epoch,
                        )
                        self.writer.add_scalars(
                            main_tag="Policy accuracy",
                            tag_scalar_dict={
                                "top1": policy_top1_acc,
                                "top3": policy_top3_acc,
                            },
                            global_step=epoch,
                        )

                else:
                    early_stop_flag = 0

                # Sync across all ranks
                should_stop_tensor = torch.tensor(early_stop_flag, device=self.rank)
                torch.distributed.broadcast(should_stop_tensor, src=0)
                should_stop = bool(should_stop_tensor.item())

                # Check if test loss is still decreasing or if user interrupted.
                if should_stop:
                    if self.rank == 0:

                        if self._interrupted:
                            logger.info(
                                f"Training interrupted by user (Ctrl+C) at epoch: {epoch+1}"
                            )
                        else:
                            logger.info(
                                f"Models test loss not decreasing significantly enough. Stopping training early at epoch: {epoch+1}"
                            )
                        logger.info(
                            f"Saving model with lowest loss from epoch: {self.early_stopping.best_score_epoch}"
                        )

                        if self.temp_checkpoint_file_path is not None:
                            os.remove(self.temp_checkpoint_file_path)

                    break

                elif self.rank == 0 and self.temp_checkpoint_file_path is not None:
                    torch.save(
                        obj=self.early_stopping.best_model_state,
                        f=self.temp_checkpoint_file_path,
                    )

                # Adjust learning rate
                if self.lr_scheduler is not None and not self.skip_lr_sched:
                    self.lr_scheduler.step()

        finally:
            signal.signal(signal.SIGINT, original_sigint_handler)

        if self.rank == 0 and self.writer:
            self.writer.close()
        if self.rank == 0 and self.temp_checkpoint_file_path is not None:
            os.remove(self.temp_checkpoint_file_path)

        return results, self.early_stopping.best_model_state

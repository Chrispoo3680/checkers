import os
import signal
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import torch
import torch.distributed as dist
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
        early_stopping_metric: str = "combined",
        lr_scheduler: Optional[Union[MultiStepLR, StepLR, CosineAnnealingLR]] = None,
        temp_checkpoint_file_path: Optional[Path] = None,
        writer: Optional[SummaryWriter] = None,
        use_ddp: bool = True,
        amp_enabled: bool = True,
        amp_dtype: torch.dtype = torch.float16,
    ):

        self.device = device
        self.rank = rank
        self.use_ddp = use_ddp
        self.amp_enabled = amp_enabled
        self.amp_dtype = amp_dtype
        inner: nn.Module = model.to(rank)
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(rank)
            if major >= 7:
                try:
                    inner = cast(nn.Module, torch.compile(inner))
                    if rank == 0:
                        logger.info(
                            "Enabled torch.compile on CUDA device capability %s.%s",
                            major,
                            minor,
                        )
                except Exception as exc:
                    if rank == 0:
                        logger.warning(
                            "torch.compile failed (%s). Falling back to eager mode.",
                            exc,
                        )
            elif rank == 0:
                logger.info(
                    "Skipping torch.compile: CUDA device capability %s.%s is below 7.0",
                    major,
                    minor,
                )
        if self.use_ddp:
            self.model: nn.Module = DDP(
                inner, device_ids=[rank], find_unused_parameters=False
            )
            self._ddp_model: Optional[DDP] = cast(DDP, self.model)
        else:
            self.model = inner
            self._ddp_model = None
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
        self.early_stopping_metric = early_stopping_metric
        self.temp_checkpoint_file_path = temp_checkpoint_file_path
        self.writer = writer
        self._interrupted = False

    def _handle_sigint(self, signum, frame):
        self._interrupted = True

    def _sync_interrupt_flag(self) -> bool:
        """Synchronize interrupt intent across all ranks."""
        interrupted = int(self._interrupted)

        if dist.is_available() and dist.is_initialized():
            interrupted_tensor = torch.tensor(interrupted, device=self.rank)
            dist.all_reduce(interrupted_tensor, op=dist.ReduceOp.MAX)
            return bool(interrupted_tensor.item())

        return bool(interrupted)

    def train_step(self, epoch: int):

        self.train_dataloader.sampler.set_epoch(epoch)  # type: ignore

        self.model.train()

        train_loss_acc = torch.zeros(1, device=self.device)
        interrupted = False

        for batch, (X, pi_target, value_target) in enumerate(
            tqdm(
                self.train_dataloader,
                position=1,
                leave=False,
                desc="Iterating through training batches",
                disable=self.rank != 0,
            )
        ):
            if batch % 50 == 0 and self._sync_interrupt_flag():
                interrupted = True
                break

            amp_context = (
                torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.amp_enabled,
                )
                if self.device.type == "cuda"
                else nullcontext()
            )
            with amp_context:
                X, pi_target, value_target = (
                    X.to(self.rank, non_blocking=True),
                    pi_target.to(self.rank, non_blocking=True),
                    value_target.to(self.rank, non_blocking=True),
                )

                policy_logist, value_pred = self.model(X)

                loss, policy_loss, value_loss, entropy = self.loss_fn(
                    policy_logist,
                    value_pred,
                    pi_target if self.train_policy else None,
                    value_target if self.train_value else None,
                    value_weight=3.0,
                )

            train_loss_acc.add_(loss.detach())

            self.optimizer.zero_grad(set_to_none=True)

            if self.amp_enabled:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
            else:
                loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            if self.amp_enabled:
                self.scaler.step(self.optimizer)
                scale = self.scaler.get_scale()
                self.scaler.update()
                self.skip_lr_sched = scale != self.scaler.get_scale()
            else:
                self.optimizer.step()
                self.skip_lr_sched = False

        return train_loss_acc.item(), interrupted

    def test_step(self, epoch: int):

        self.test_dataloader.sampler.set_epoch(epoch)  # type: ignore

        self.model.eval()

        test_loss_acc = torch.zeros(1, device=self.device)
        value_preds_list = []
        value_targets_list = []
        policy_top1_correct = 0
        policy_top1_total = 0
        policy_top3_correct = 0
        policy_top3_total = 0
        interrupted = False

        amp_context = (
            torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp_enabled,
            )
            if self.device.type == "cuda"
            else nullcontext()
        )
        with amp_context:
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
                    if batch % 50 == 0 and self._sync_interrupt_flag():
                        interrupted = True
                        break

                    X, pi_target, value_target = (
                        X.to(self.rank, non_blocking=True),
                        pi_target.to(self.rank, non_blocking=True),
                        value_target.to(self.rank, non_blocking=True),
                    )

                    policy_logist, value_pred = self.model(X)

                    if self.train_value:
                        value_preds_list.append(value_pred.detach())
                        value_targets_list.append(value_target.detach())

                    if self.train_policy:
                        legal_mask = pi_target > 0
                        batch_correct, batch_total = utils.policy_top1_accuracy(
                            policy_logist, pi_target, legal_mask
                        )
                        policy_top1_correct += batch_correct
                        policy_top1_total += batch_total
                        batch_top3_correct, batch_top3_total = (
                            utils.policy_top3_accuracy(
                                policy_logist, pi_target, legal_mask
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

                    test_loss_acc.add_(loss)

        if interrupted:
            return test_loss_acc.item(), 0.0, 0.0, 0.0, True

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

        return (
            test_loss_acc.item(),
            value_score,
            policy_top1_acc,
            policy_top3_acc,
            False,
        )

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
                (
                    train_loss,
                    interrupted,
                ) = self.train_step(epoch)
                if interrupted:
                    test_loss, value_score, policy_top1_acc, policy_top3_acc = (
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    )
                else:
                    (
                        test_loss,
                        value_score,
                        policy_top1_acc,
                        policy_top3_acc,
                        interrupted,
                    ) = self.test_step(epoch)

                if not interrupted:
                    results["learning_rate"].append(
                        self.optimizer.param_groups[0]["lr"]
                    )
                    results["train_loss"].append(train_loss)
                    results["test_loss"].append(test_loss)
                    results["value_score"].append(value_score)
                    results["policy_top1_accuracy"].append(policy_top1_acc)
                    results["policy_top3_accuracy"].append(policy_top3_acc)

                if self.rank == 0:
                    global_interrupted = interrupted or self._sync_interrupt_flag()

                    if not global_interrupted:
                        if self.early_stopping_metric == "policy":
                            es_score = policy_top1_acc
                        elif self.early_stopping_metric == "value":
                            es_score = value_score
                        else:  # "combined"
                            es_score = 0.5 * value_score + 0.5 * policy_top1_acc
                        best_model_ref = (
                            self._ddp_model.module
                            if self._ddp_model is not None
                            else self.model
                        )
                        self.early_stopping(es_score, best_model_ref, epoch + 1)

                    early_stop_flag = int(
                        self.early_stopping.early_stop or global_interrupted
                    )

                    # Log and save epoch loss and accuracy results
                    if not global_interrupted:
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
                    if self.writer and not global_interrupted:
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
                    global_interrupted = interrupted or self._sync_interrupt_flag()
                    early_stop_flag = int(global_interrupted)

                # Sync across all ranks only when DDP is active.
                if dist.is_available() and dist.is_initialized():
                    should_stop_tensor = torch.tensor(early_stop_flag, device=self.rank)
                    torch.distributed.broadcast(should_stop_tensor, src=0)
                    should_stop = bool(should_stop_tensor.item())
                else:
                    should_stop = bool(early_stop_flag)

                # Check if test loss is still decreasing or if user interrupted.
                if should_stop:
                    if self.rank == 0 and epoch > 0:

                        if interrupted or self._interrupted:
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
                            if self.temp_checkpoint_file_path.exists():
                                os.remove(self.temp_checkpoint_file_path)

                    break

                elif self.rank == 0 and self.temp_checkpoint_file_path is not None:
                    torch.save(
                        obj=self.early_stopping.best_model_state,
                        f=self.temp_checkpoint_file_path,
                    )

                # Adjust learning rate
                if (
                    self.lr_scheduler is not None
                    and not self.skip_lr_sched
                    and not interrupted
                ):
                    self.lr_scheduler.step()

        finally:
            signal.signal(signal.SIGINT, original_sigint_handler)

        if self.rank == 0 and self.writer:
            self.writer.close()
        if self.rank == 0 and self.temp_checkpoint_file_path is not None:
            if self.temp_checkpoint_file_path.exists():
                os.remove(self.temp_checkpoint_file_path)

        return results, self.early_stopping.best_model_state

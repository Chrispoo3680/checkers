import torch
from torch import nn
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.nn.parallel import DistributedDataParallel as DDP

from pathlib import Path
import os
from tqdm import tqdm

from ..common import tools, utils


from typing import Dict, List, Optional, Any, Union, Tuple


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
        loss_fn: nn.Module,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        device: torch.device,
        rank: int,
        scaler: GradScaler,
        early_stopping: utils.EarlyStopping,
        lr_scheduler: Optional[Union[MultiStepLR, StepLR]] = None,
        temp_checkpoint_file_path: Optional[Path] = None,
        writer: Optional[SummaryWriter] = None,
    ):

        self.device = device
        self.rank = rank
        self.model = DDP(model.to(rank), device_ids=[rank])
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.scaler = scaler
        self.skip_lr_sched = False
        self.lr_scheduler = lr_scheduler
        self.early_stopping = early_stopping
        self.temp_checkpoint_file_path = temp_checkpoint_file_path
        self.writer = writer

    def train_step(self, epoch: int):

        self.train_dataloader.sampler.set_epoch(epoch)  # type: ignore

        self.model.train()

        train_loss, train_acc = 0, 0

        for batch, (X, y) in enumerate(
            tqdm(
                self.train_dataloader,
                position=1,
                leave=False,
                desc="Iterating through training batches.",
                disable=self.rank != 0,
            )
        ):
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                X, y = X.to(self.rank), y.to(self.rank)
                y_pred = self.model(X)
                loss = self.loss_fn(y_pred, y)
                train_loss += loss.item()

            self.scaler.scale(loss).backward()

            self.scaler.step(self.optimizer)

            scale = self.scaler.get_scale()
            self.scaler.update()
            self.skip_lr_sched = scale != self.scaler.get_scale()

            self.optimizer.zero_grad(set_to_none=True)

            # Calculate and accumulate accuracy metric across all batches
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item() / len(y_pred)

        # Adjust metrics to get average loss and accuracy per batch
        train_loss = train_loss / len(self.train_dataloader)
        train_acc = train_acc / len(self.train_dataloader)
        return train_loss, train_acc

    def test_step(self, epoch: int):

        self.test_dataloader.sampler.set_epoch(epoch)  # type: ignore

        self.model.eval()

        test_loss, test_acc = 0, 0

        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            with torch.inference_mode():
                for batch, (X, y) in enumerate(
                    tqdm(
                        self.test_dataloader,
                        position=1,
                        leave=False,
                        desc="Iterating through testing batches.",
                        disable=self.rank != 0,
                    )
                ):
                    X, y = X.to(self.rank), y.to(self.rank)

                    test_pred_logits = self.model(X)

                    loss = self.loss_fn(test_pred_logits, y)
                    test_loss += loss.item()

                    # Calculate and accumulate accuracy
                    test_pred_labels = test_pred_logits.argmax(dim=1)
                    test_acc += (test_pred_labels == y).sum().item() / len(
                        test_pred_labels
                    )

        # Adjust metrics to get average loss and accuracy per batch
        test_loss = test_loss / len(self.test_dataloader)
        test_acc = test_acc / len(self.test_dataloader)
        return test_loss, test_acc

    def train(self, epochs: int) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:

        results: Dict[str, List[float]] = {
            "learning_rate": [],
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": [],
        }

        for epoch in tqdm(
            range(epochs),
            position=0,
            desc="Iterating through epochs.",
            disable=self.rank != 0,
        ):
            train_loss, train_acc = self.train_step(epoch)
            test_loss, test_acc = self.test_step(epoch)

            results["learning_rate"].append(self.optimizer.param_groups[0]["lr"])
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)

            self.early_stopping(test_loss, self.model.module, epoch + 1)

            if self.rank == 0:

                early_stop_flag = int(self.early_stopping.early_stop)

                # Log and save epoch loss and accuracy results
                logger.info(
                    f"      GPU ID: {self.rank}  |  "
                    f"epoch: {epoch+1}  |  "
                    f"train_loss: {train_loss:.4f}  |  "
                    f"train_acc: {train_acc:.4f}  |  "
                    f"test_loss: {test_loss:.4f}  |  "
                    f"test_acc: {test_acc:.4f}  |  "
                    f"learning_rate: {self.optimizer.param_groups[0]['lr']}  |  "
                    f"early stopping counter: {self.early_stopping.counter} / {self.early_stopping.patience}"
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
                    self.writer.add_scalars(
                        main_tag="Accuracy",
                        tag_scalar_dict={"train_acc": train_acc, "test_acc": test_acc},
                        global_step=epoch,
                    )

                    self.writer.close()

            else:
                early_stop_flag = 0

            # Sync across all ranks
            should_stop_tensor = torch.tensor(early_stop_flag, device=self.rank)
            torch.distributed.broadcast(should_stop_tensor, src=0)
            should_stop = bool(should_stop_tensor.item())

            # Check if test loss is still decreasing. If not decreasing for multiple epochs, break the loop
            if should_stop:
                if self.rank == 0:

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

        return results, self.early_stopping.best_model_state

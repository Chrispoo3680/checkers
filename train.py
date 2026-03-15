"""
This is a file for training the lego object detection model.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.multiprocessing as mp
from torch.amp.grad_scaler import GradScaler

repo_root_dir: Path = Path(__file__).parent
sys.path.append(str(repo_root_dir))

from src.common import tools, utils
from src.data import download
from src.model import models
from src.model.trainer import Trainer
from src.preprocess import build_features


def main(
    rank: int,
    world_size: int,
    NUM_EPOCHS,
    BATCH_SIZE,
    NUM_WORKERS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    LR_STEP_INTERVAL,
    FROZEN_BLOCKS,
    CHECKPOINT_PATH,
    MODEL_NAME,
    FREEZE_BODY,
    FREEZE_POLICY_HEAD,
    FREEZE_VALUE_HEAD,
    TRAIN_POLICY,
    TRAIN_VALUE,
    EARLY_STOPPING_METRIC,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_DELTA,
    data_paths,
    temp_checkpoint_dir,
    model_save_path,
    model_save_name_version,
    results_save_path,
    writer_config,  # Changed: pass config dict instead of writer object
    device_type,  # Changed: pass device type string instead of device object
):

    use_ddp = world_size > 1

    if use_ddp:
        utils.ddp_setup(rank, world_size)

    try:
        if rank == 0:
            try:
                logging_file_path = os.environ["LOGGING_FILE_PATH"]
            except KeyError:
                logging_file_path = None

            logger = tools.create_logger(
                log_path=logging_file_path, logger_name=__name__
            )

        else:
            logger = logging.getLogger("silent_logger")
            logger.setLevel(logging.INFO)
            logger.addHandler(logging.NullHandler())

        # Create device per worker
        device = torch.device(f"cuda:{rank}" if device_type == "cuda" else "cpu")

        if device_type == "cuda":
            torch.backends.cudnn.benchmark = True
            if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")

        amp_enabled = device_type == "cuda"
        amp_dtype = torch.float16
        if device_type == "cuda":
            major, minor = torch.cuda.get_device_capability(rank)
            # GTX 10xx (sm_6.x) has no Tensor Cores; AMP can be slower there.
            if major < 7:
                amp_enabled = False
                logger.info(
                    "Disabling AMP on CUDA capability %s.%s (no Tensor Cores).",
                    major,
                    minor,
                )
            else:
                logger.info(
                    "Enabling AMP (float16) on CUDA capability %s.%s.", major, minor
                )

        # Create writer per worker (only for rank 0)
        if rank == 0 and writer_config is not None:
            writer = utils.create_writer(
                root_dir=writer_config["root_dir"],
                experiment_name=writer_config["experiment_name"],
                model_name=writer_config["model_name"],
                var=writer_config["var"],
            )
        else:
            writer = None

        # Create the classification model
        logger.info("Loading model...")

        model = models.CheckersNetV3(
            input_planes=18,  # 12 piece planes + side-to-move + 4 castling + en-passant
            policy_planes=73,
            channels=256,
            num_blocks=15,
            se_every_n_blocks=4,
            temperature=1.0,
        )

        if CHECKPOINT_PATH:
            logger.info(f"Loading model weights from checkpoint: {CHECKPOINT_PATH}...")
            model.load_state_dict(torch.load(CHECKPOINT_PATH))

        # Freeze entire body if requested (initial conv + all residual blocks)
        if FREEZE_BODY:
            logger.info(
                "Freezing entire model body (initial conv + all residual blocks)..."
            )
            for param in model.conv_input.parameters():
                param.requires_grad = False
            for param in model.bn_input.parameters():
                param.requires_grad = False
            for block in model.res_blocks:
                for param in block.parameters():
                    param.requires_grad = False

        # Freeze specific residual blocks if requested
        elif FROZEN_BLOCKS:
            logger.info(f"Freezing specific residual blocks: {FROZEN_BLOCKS}")
            for block_idx in FROZEN_BLOCKS:
                if 0 <= block_idx < len(model.res_blocks):
                    for param in model.res_blocks[block_idx].parameters():
                        param.requires_grad = False
                    logger.info(f"  - Frozen block {block_idx}")
                else:
                    logger.warning(
                        f"  - Block index {block_idx} out of range (0-{len(model.res_blocks)-1}), skipping"
                    )

        # Freeze policy head if requested or if its loss is disabled (avoids DDP unused-param overhead)
        if FREEZE_POLICY_HEAD or not TRAIN_POLICY:
            if not FREEZE_POLICY_HEAD:
                logger.info(
                    "Policy loss disabled — freezing policy head to avoid DDP overhead."
                )
            else:
                logger.info("Freezing policy head...")
            for module in model.policy_head_modules():
                for param in module.parameters():
                    param.requires_grad = False

        # Freeze value head if requested or if its loss is disabled
        if FREEZE_VALUE_HEAD or not TRAIN_VALUE:
            if not FREEZE_VALUE_HEAD:
                logger.info(
                    "Value loss disabled — freezing value head to avoid DDP overhead."
                )
            else:
                logger.info("Freezing value head...")
            for module in model.value_head_modules():
                for param in module.parameters():
                    param.requires_grad = False

        # Create train and test dataloaders
        logger.info("Creating dataloaders...")

        cache_decode_dtype = torch.float16 if amp_enabled else torch.float32
        logger.info("Cache decode dtype set to %s", cache_decode_dtype)

        # Create train/test dataloader
        # Use num_workers=0 to avoid spawning additional processes (critical for memory)
        (train_dataloader, test_dataloader), _ = build_features.create_dataloaders(
            data_dir_paths=data_paths,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            cache_decode_dtype=cache_decode_dtype,
        )

        logger.info("Successfully created dataloaders.")

        # Set loss, optimizer and learning rate scheduling
        loss_fn = utils.chess_loss

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=NUM_EPOCHS,
            eta_min=1e-6,
        )

        # Train model with the training loop
        logger.info("Starting training...\n")

        early_stopping = utils.EarlyStopping(
            patience=EARLY_STOPPING_PATIENCE,
            delta=EARLY_STOPPING_DELTA,
        )

        # Set up scaler for mixed precision efficiency
        scaler = GradScaler(enabled=amp_enabled)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_policy=TRAIN_POLICY,
            train_value=TRAIN_VALUE,
            early_stopping_metric=EARLY_STOPPING_METRIC,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            device=device,
            rank=rank,
            scaler=scaler,
            early_stopping=early_stopping,
            lr_scheduler=lr_scheduler,
            temp_checkpoint_file_path=temp_checkpoint_dir
            / (model_save_name_version + ".pt"),
            writer=writer,
            use_ddp=use_ddp,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )

        results, best_state = trainer.train(NUM_EPOCHS)

        if rank == 0 and best_state:
            # Save the trained model
            utils.save_model(
                model=best_state,
                target_dir_path=model_save_path,
                model_name=model_save_name_version + ".pt",
            )

            # Save training results
            results_json = json.dumps(results, indent=4)

            with open(
                results_save_path / (model_save_name_version + "_results.json"), "w"
            ) as f:
                f.write(results_json)
    finally:
        if use_ddp:
            utils.ddp_cleanup()


if __name__ == "__main__":

    # Setup arguments parsing for hyperparameters
    parser = argparse.ArgumentParser(description="Hyperparameter configuration")

    parser.add_argument("--num-epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes per training process",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.0001, help="Weight decay"
    )
    parser.add_argument(
        "--lr-step-interval",
        type=int,
        default=10,
        help="Step interval for the learning rate",
    )
    parser.add_argument(
        "--frozen-blocks",
        type=str,
        default="",
        help="Comma-separated indices of specific residual blocks to freeze (e.g., '0,1,2'). Cannot be used with --freeze-body.",
    )
    parser.add_argument(
        "--freeze-body",
        action="store_true",
        help="Freeze the entire model body (initial convolution + all residual blocks). Cannot be used with --frozen-blocks.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="",
        help="Path to checkpoint used to initialize model weights",
    )
    parser.add_argument(
        "--specified-datasets",
        type=str,
        default="",
        help="Folder names of specific datasets to use for training, separated by commas (e.g., 'dataset1,dataset2'). If not specified, all datasets in the data directory will be used.",
    )
    parser.add_argument(
        "--model-name", type=str, required=True, help="Loaded models name"
    )
    parser.add_argument(
        "--freeze-policy-head",
        action="store_true",
        help="Freeze policy head during training",
    )
    parser.add_argument(
        "--freeze-value-head",
        action="store_true",
        help="Freeze value head during training",
    )
    parser.add_argument(
        "--train-policy",
        action="store_true",
        help="Whether to compute and use policy loss during training (default: True). Cannot be True if --freeze-policy-head is set.",
    )
    parser.add_argument(
        "--train-value",
        action="store_true",
        help="Whether to compute and use value loss during training (default: True). Cannot be True if --freeze-value-head is set.",
    )
    parser.add_argument(
        "--early-stopping-metric",
        type=str,
        default="combined",
        choices=["combined", "policy", "value"],
        help="Metric used for early stopping: 'combined' (0.5*value_score + 0.5*policy_top1), 'policy' (policy_top1_acc), or 'value' (value_score)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Number of epochs without improvement before stopping early",
    )
    parser.add_argument(
        "--early-stopping-delta",
        type=float,
        default=0.001,
        help="Minimum improvement in the early stopping metric to count as progress",
    )
    parser.add_argument(
        "--experiment-name", type=str, default=None, help="Experiment name"
    )
    parser.add_argument(
        "--experiment-variable", type=str, default=None, help="Experiment variable"
    )

    parser.add_argument(
        "--model-save-name",
        type=str,
        default=parser.parse_known_args()[0].model_name,
        help="Model save name",
    )

    args = parser.parse_args()

    # Setup hyperparameters
    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    LEARNING_RATE = args.learning_rate
    WEIGHT_DECAY = args.weight_decay
    LR_STEP_INTERVAL = args.lr_step_interval
    FROZEN_BLOCKS = [int(b) for b in args.frozen_blocks.split(",") if b != ""]
    FREEZE_BODY = args.freeze_body
    CHECKPOINT_PATH = args.checkpoint_path
    SPECIFIED_DATASETS = (
        [name.strip() for name in args.specified_datasets.split(",")]
        if args.specified_datasets
        else []
    )
    MODEL_NAME = args.model_name
    FREEZE_POLICY_HEAD = args.freeze_policy_head
    FREEZE_VALUE_HEAD = args.freeze_value_head
    TRAIN_POLICY = args.train_policy
    TRAIN_VALUE = args.train_value
    EARLY_STOPPING_METRIC = args.early_stopping_metric
    EARLY_STOPPING_PATIENCE = args.early_stopping_patience
    EARLY_STOPPING_DELTA = args.early_stopping_delta
    EXPERIMENT_NAME = args.experiment_name
    EXPERIMENT_VARIABLE = args.experiment_variable

    # Validate that freeze-body and frozen-blocks are not used together
    if FREEZE_BODY and FROZEN_BLOCKS:
        raise ValueError(
            "Cannot use both --freeze-body and --frozen-blocks. Choose one."
        )

    # Default: train a head if neither its loss nor its freeze flag was explicitly set
    if not TRAIN_POLICY and not TRAIN_VALUE:
        TRAIN_POLICY = not FREEZE_POLICY_HEAD
        TRAIN_VALUE = not FREEZE_VALUE_HEAD

    # Validate that frozen heads cannot be trained
    if FREEZE_POLICY_HEAD and TRAIN_POLICY:
        raise ValueError(
            "Cannot train policy head when it is frozen. Either remove --freeze-policy-head or remove --train-policy."
        )
    if FREEZE_VALUE_HEAD and TRAIN_VALUE:
        raise ValueError(
            "Cannot train value head when it is frozen. Either remove --freeze-value-head or remove --train-value."
        )

    config = tools.load_config()

    # Setup directories
    data_path: Path = repo_root_dir / config["data_path"]
    os.makedirs(data_path, exist_ok=True)

    model_save_path: Path = repo_root_dir / config["model_path"]
    os.makedirs(model_save_path, exist_ok=True)
    model_save_name_version: str = utils.model_save_version(
        save_dir_path=model_save_path, save_name=MODEL_NAME
    )

    temp_checkpoint_dir: Path = repo_root_dir / config["temp_checkpoint_path"]
    os.makedirs(temp_checkpoint_dir, exist_ok=True)

    results_save_path: Path = repo_root_dir / config["results_path"]
    os.makedirs(results_save_path, exist_ok=True)

    logging_dir_path: Path = repo_root_dir / config["logging_path"]
    os.makedirs(logging_dir_path, exist_ok=True)

    logging_file_path: Path = logging_dir_path / (
        model_save_name_version + "_training.log"
    )
    os.environ["LOGGING_FILE_PATH"] = str(logging_file_path)

    # Setup logging for info and debugging
    logger: logging.Logger = tools.create_logger(
        log_path=logging_file_path, logger_name=__name__
    )
    logger.info("\n\n")
    logger.info(f"Logging to file: {logging_file_path}")

    # Setup SummaryWriter for tensorboards
    if EXPERIMENT_NAME and EXPERIMENT_VARIABLE:
        create_writer: bool = True
        logger.info(
            f"Experiment name: {EXPERIMENT_NAME}, Experiment variable: {EXPERIMENT_VARIABLE}. Will create SummaryWriter and save tensorboard logs to runs/classification/{EXPERIMENT_NAME}/{MODEL_NAME}/{EXPERIMENT_VARIABLE}..."
        )
    elif EXPERIMENT_NAME or EXPERIMENT_VARIABLE:
        raise NameError(
            "You need to apply a string value to both '--experiment_name' and '--experiment_variable' to use either."
        )
    else:
        create_writer = False

    # Download dataset if not already downloaded
    if os.listdir(data_path):
        logger.info(
            f"There already exists files in directory: {data_path}. Assuming datasets are already downloaded!"
        )
    else:
        download.api_scraper_download_data(
            download_url=config["lichess_db_eval_dataset_download_url"],
            save_path=data_path,
            data_name=config["lichess_db_eval_dataset_name"],
        )

        download.kaggle_download_data(
            data_handle=config["lichess_puzzles_dataset_handle"],
            save_path=data_path,
            data_name=config["lichess_puzzles_dataset_name"],
        )
        download.kaggle_download_data(
            data_handle=config["magnus_carlsen_games_dataset_handle"],
            save_path=data_path,
            data_name=config["magnus_carlsen_games_dataset_name"],
        )
        download.kaggle_download_data(
            data_handle=config["stockfish_position_evaluations_dataset_handle"],
            save_path=data_path,
            data_name=config["stockfish_position_evaluations_dataset_name"],
        )

    file_extentions = [".parquet", ".csv", ".jsonl"]

    # Finding all paths to image data in downloaded datasets
    if SPECIFIED_DATASETS:
        logger.info(
            f"Specified datasets provided: {SPECIFIED_DATASETS}. Only using data from these folders for training."
        )
        data_paths: list[Path] = []
        for dataset_name in SPECIFIED_DATASETS:
            dataset_folder_path = data_path / dataset_name
            if not dataset_folder_path.exists():
                logger.warning(
                    f"Specified dataset folder does not exist: {dataset_folder_path}. Skipping this dataset."
                )
                continue

            data_paths.extend(
                tools.get_files_from_folder(
                    root_folder_path=dataset_folder_path, extensions=file_extentions
                )
            )

        if not data_paths:
            raise ValueError(
                "No valid datasets found based on the specified dataset names. Please check the provided names and ensure the corresponding folders and .parquet files exist."
            )
    else:
        logger.info(
            f"No specified datasets provided. Using all datasets in data directory: {data_path} for training."
        )
        data_paths: list[Path] = tools.get_files_from_folder(
            root_folder_path=data_path, extensions=file_extentions
        )

    print(data_paths)

    # Setup target device
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device = {device_type}")

    world_size = torch.cuda.device_count() if device_type == "cuda" else 1

    # Prepare writer config (pass as dict, not the object itself)
    if create_writer:
        writer_config = {
            "root_dir": repo_root_dir,
            "experiment_name": EXPERIMENT_NAME,
            "model_name": model_save_name_version,
            "var": EXPERIMENT_VARIABLE,
        }
    else:
        writer_config = None

    # Logging hyperparameters
    logger.info(
        f"Using hyperparameters:"
        f"\n    num_epochs = {NUM_EPOCHS}"
        f"\n    batch_size = {BATCH_SIZE}"
        f"\n    num_workers = {NUM_WORKERS}"
        f"\n    learning_rate = {LEARNING_RATE}"
        f"\n    weight_decay = {WEIGHT_DECAY}"
        f"\n    lr_step_interval = {LR_STEP_INTERVAL}"
        f"\n    frozen_blocks = {FROZEN_BLOCKS}"
        f"\n    freeze_body = {FREEZE_BODY}"
        f"\n    checkpoint_path = {CHECKPOINT_PATH}"
        f"\n    specified_datasets = {SPECIFIED_DATASETS if SPECIFIED_DATASETS else 'All datasets in data directory'}"
        f"\n    model_name = {MODEL_NAME}"
        f"\n    model_save_name = {model_save_name_version}"
        f"\n    freeze_policy_head = {FREEZE_POLICY_HEAD}"
        f"\n    freeze_value_head = {FREEZE_VALUE_HEAD}"
        f"\n    train_policy = {TRAIN_POLICY}"
        f"\n    train_value = {TRAIN_VALUE}"
        f"\n    early_stopping_metric = {EARLY_STOPPING_METRIC}"
        f"\n    early_stopping_patience = {EARLY_STOPPING_PATIENCE}"
        f"\n    early_stopping_delta = {EARLY_STOPPING_DELTA}"
        f"\n    experiment_name = {EXPERIMENT_NAME}"
        f"\n    experiment_name = {EXPERIMENT_VARIABLE}"
        f"\n    world_size = {world_size}"
    )

    try:
        if world_size == 1:
            main(
                rank=0,
                world_size=1,
                NUM_EPOCHS=NUM_EPOCHS,
                BATCH_SIZE=BATCH_SIZE,
                NUM_WORKERS=NUM_WORKERS,
                LEARNING_RATE=LEARNING_RATE,
                WEIGHT_DECAY=WEIGHT_DECAY,
                LR_STEP_INTERVAL=LR_STEP_INTERVAL,
                FROZEN_BLOCKS=FROZEN_BLOCKS,
                CHECKPOINT_PATH=CHECKPOINT_PATH,
                MODEL_NAME=MODEL_NAME,
                FREEZE_BODY=FREEZE_BODY,
                FREEZE_POLICY_HEAD=FREEZE_POLICY_HEAD,
                FREEZE_VALUE_HEAD=FREEZE_VALUE_HEAD,
                TRAIN_POLICY=TRAIN_POLICY,
                TRAIN_VALUE=TRAIN_VALUE,
                EARLY_STOPPING_METRIC=EARLY_STOPPING_METRIC,
                EARLY_STOPPING_PATIENCE=EARLY_STOPPING_PATIENCE,
                EARLY_STOPPING_DELTA=EARLY_STOPPING_DELTA,
                data_paths=data_paths,
                temp_checkpoint_dir=temp_checkpoint_dir,
                model_save_path=model_save_path,
                model_save_name_version=model_save_name_version,
                results_save_path=results_save_path,
                writer_config=writer_config,
                device_type=device_type,
            )
        else:
            mp.spawn(  # type: ignore
                main,
                args=(
                    world_size,
                    NUM_EPOCHS,
                    BATCH_SIZE,
                    NUM_WORKERS,
                    LEARNING_RATE,
                    WEIGHT_DECAY,
                    LR_STEP_INTERVAL,
                    FROZEN_BLOCKS,
                    CHECKPOINT_PATH,
                    MODEL_NAME,
                    FREEZE_BODY,
                    FREEZE_POLICY_HEAD,
                    FREEZE_VALUE_HEAD,
                    TRAIN_POLICY,
                    TRAIN_VALUE,
                    EARLY_STOPPING_METRIC,
                    EARLY_STOPPING_PATIENCE,
                    EARLY_STOPPING_DELTA,
                    data_paths,
                    temp_checkpoint_dir,
                    model_save_path,
                    model_save_name_version,
                    results_save_path,
                    writer_config,  # Pass config dict instead of writer object
                    device_type,  # Pass device type string instead of device object
                ),
                nprocs=world_size,
            )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user (Ctrl+C). Exiting gracefully.")

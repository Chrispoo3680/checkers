import yaml
from tqdm import tqdm
import logging

import os
import sys
from pathlib import Path
from zipfile import ZipFile
import pandas as pd
from contextlib import contextmanager

repo_root_dir: Path = Path(__file__).parent.parent.parent
sys.path.append(str(repo_root_dir))


from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record) -> None:
        msg: str = self.format(record)
        tqdm.write(msg)


def create_logger(
    logger_name: str, log_path: Optional[Union[str, Path]] = None
) -> logging.Logger:

    config = load_config()

    level_dict = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }

    # Set up logging
    logger = logging.getLogger(logger_name)
    logger.setLevel(level_dict[config["logging_level"]])

    if logger.hasHandlers():
        logger.handlers.clear()

    # Tqdm handler for terminal output
    tqdm_handler = TqdmLoggingHandler()
    tqdm_handler.setFormatter(
        logging.Formatter("%(asctime)s  -  %(name)s  -  %(levelname)s:    %(message)s")
    )
    logger.addHandler(tqdm_handler)

    # File handler for .log file output
    if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s  -  %(name)s  -  %(levelname)s:    %(message)s"
            )
        )
        logger.addHandler(file_handler)

    return logger


def load_config():
    # Read in the configuration file
    with open(repo_root_dir / "config.yaml") as p:
        config = yaml.safe_load(p)
    return config


def rename_and_unzip_file(
    zip_file_path: Union[str, Path], new_file_path: Union[str, Path]
) -> None:
    with ZipFile(zip_file_path, "r") as zipped:
        zipped.extractall(path=new_file_path)

    os.remove(zip_file_path)


def get_part_cat(part_id: str, id_to_cat: Dict[str, int]) -> int:

    # Predefine possible suffixes
    possible_letters = ["", "a", "b", "c"]

    # Split the part ID into slices
    part_slices = part_id.split("_")

    for part_slice in part_slices:
        # Extract the numeric part and suffix
        part_num, part_letter = (
            (part_slice[:-1], part_slice[-1])
            if part_slice[-1] in possible_letters
            else (part_slice, "")
        )

        # Generate possible part IDs to check
        candidate_ids = [
            part_num + letter
            for letter in [part_letter]
            + [l for l in possible_letters if l != part_letter]
        ]

        for candidate_id in candidate_ids:
            if candidate_id in id_to_cat:
                return id_to_cat[candidate_id]

    logging.error(f"Couldn't find any part categories for part with id: {part_id}\n\n")
    return 0


def part_cat_csv_to_dict(part_to_cat_path: Union[str, Path]) -> Dict[str, int]:
    part_df = pd.read_csv(part_to_cat_path, sep=",")

    part_nums = part_df["part_num"].to_numpy()
    part_cat_ids = part_df["part_cat_id"].to_numpy()

    num_to_cat: Dict[str, int] = {num: cat for num, cat in zip(part_nums, part_cat_ids)}

    return num_to_cat


def read_file(path) -> str:
    with open(path, "r") as f:
        return f.read()


@contextmanager
def suppress_stderr():
    stderr_fd = sys.stderr.fileno()
    # Save a copy of the original stderr
    with os.fdopen(os.dup(stderr_fd), "w") as old_stderr:
        # Redirect stderr to /dev/null
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
        try:
            yield
        finally:
            # Restore stderr
            os.dup2(old_stderr.fileno(), stderr_fd)

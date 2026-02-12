import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union
from zipfile import ZipFile

import torch
import yaml
from tqdm import tqdm

repo_root_dir: Path = Path(__file__).parent.parent.parent
sys.path.append(str(repo_root_dir))

PROMOTION_MAP = {None: 0, "n": 1, "b": 2, "r": 3, "q": 4}
INDEX_TO_PROMO = {v: k for k, v in PROMOTION_MAP.items()}


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


def decode_fen_pos(fen: str) -> torch.Tensor:
    """
    Decodes a FEN string into a torch.tuple with size (12, 8, 8),
    where the channels is the different piece types.
    """

    pieces = "rnbqkpRNBQKP"

    decoded_board = torch.zeros((12, 8, 8), dtype=torch.float32)

    for row_idx, row in enumerate(fen.split("/")):
        col_idx = 0
        for char in row:
            if char.isdigit():
                col_idx += int(char)  # Skip empty squares
            else:
                piece_idx = pieces.index(char)
                decoded_board[piece_idx, row_idx, col_idx] = 1.0
                col_idx += 1

    return decoded_board


def square_to_index(square):
    file = ord(square[0]) - ord("a")
    rank = int(square[1]) - 1
    return rank * 8 + file


def index_to_square(index):
    file = index % 8
    rank = index // 8
    return chr(file + ord("a")) + str(rank + 1)


def encode_USI_to_int(move_uci):
    """
    move_uci example:
    "e2e4"
    "e7e8q"
    """

    from_sq = square_to_index(move_uci[:2])
    to_sq = square_to_index(move_uci[2:4])

    promotion = move_uci[4] if len(move_uci) == 5 else None
    promo_id = PROMOTION_MAP[promotion]

    index = from_sq * 64 * 5 + to_sq * 5 + promo_id
    return index


def decode_int_to_UCI(index):

    from_sq = index // (64 * 5)
    remainder = index % (64 * 5)

    to_sq = remainder // 5
    promo_id = remainder % 5

    from_square = index_to_square(from_sq)
    to_square = index_to_square(to_sq)

    promotion = INDEX_TO_PROMO[promo_id]

    if promotion is None:
        return from_square + to_square
    else:
        return from_square + to_square + promotion


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

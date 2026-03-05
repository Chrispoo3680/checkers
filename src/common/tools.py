import io
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union
from zipfile import ZipFile

import chess
import chess.pgn
import pandas as pd
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


def get_files_from_folder(root_folder_path: Union[str, Path], extension: str):
    data_paths: list[Path] = []
    for root, _, files in os.walk(root_folder_path):
        for file_name in files:
            if file_name.endswith(extension):
                data_paths.append(Path(root) / file_name)
    return data_paths


def encode_fen_pos(fen: str) -> torch.Tensor:
    """
    Encode FEN into tensor (18, 8, 8)
    """

    board_fen, side, castling, ep_square, _, _ = fen.split(" ")

    pieces = "RNBQKPrnbqkp"

    encoded_board = torch.zeros((18, 8, 8), dtype=torch.float32)

    # ---------------- PIECES ----------------
    for row_idx, row in enumerate(board_fen.split("/")):
        col_idx = 0

        for char in row:
            if char.isdigit():
                col_idx += int(char)
            else:
                piece_idx = pieces.index(char)
                encoded_board[piece_idx, row_idx, col_idx] = 1.0
                col_idx += 1

        # Defensive sanity check (VERY useful)
        assert col_idx == 8

    # ---------------- SIDE TO MOVE ----------------
    if side == "w":
        encoded_board[12, :, :] = 1.0

    # ---------------- CASTLING RIGHTS ----------------
    if "K" in castling:
        encoded_board[13, :, :] = 1.0

    if "Q" in castling:
        encoded_board[14, :, :] = 1.0

    if "k" in castling:
        encoded_board[15, :, :] = 1.0

    if "q" in castling:
        encoded_board[16, :, :] = 1.0

    # ---------------- EN PASSANT ----------------
    if ep_square != "-":
        ep_idx = square_to_index(ep_square)

        row = ep_idx // 8
        col = ep_idx % 8

        encoded_board[17, row, col] = 1.0

    return encoded_board


def square_to_index(square):
    file = ord(square[0]) - ord("a")
    rank = int(square[1]) - 1

    rank = 7 - rank

    return rank * 8 + file


def index_to_square(index):
    file = index % 8
    rank = index // 8

    rank = 7 - rank

    return chr(file + ord("a")) + str(rank + 1)


def encode_UCI_to_int(move_uci):

    if not isinstance(move_uci, str):
        raise ValueError(f"Invalid move type: {move_uci} ({type(move_uci)})")

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


def expand_game_positions(puzzles_df, fen_col="FEN", moves_col="Moves"):
    """
    Expand chess games into individual position-move pairs.

    Parameters:
    -----------
    puzzles_df : pd.DataFrame
        DataFrame containing chess games with FEN positions and move sequences
    fen_col : str
        Name of the column containing FEN positions (default: 'FEN')
    moves_col : str
        Name of the column containing move sequences (default: 'Moves')

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns 'fen' and 'move', one row per move in each game
    """
    expanded_data = []

    for _, row in tqdm(puzzles_df.iterrows()):
        board = chess.Board(row[fen_col])

        # Split moves if it's a string, otherwise assume it's already a list
        if isinstance(row[moves_col], str):
            moves = row[moves_col].split()
        else:
            moves = row[moves_col]

        for move_uci in moves:
            # Store current position and move
            expanded_data.append({"fen": board.fen(), "move": move_uci})

            # Make the move to get to next position
            try:
                move = chess.Move.from_uci(move_uci)
                board.push(move)
            except ValueError:
                # Skip invalid moves
                continue

    return pd.DataFrame(expanded_data)


def _StrictGameBuilder():
    """GameBuilder that flags games containing illegal moves."""

    class Builder(chess.pgn.GameBuilder):
        def __init__(self):
            super().__init__()
            self.has_error = False

        def handle_error(self, error):
            self.has_error = True

    return Builder()


def expand_game_positions_san(games_df, moves_col="Moves"):
    """
    Expand chess games into individual position-move pairs, where moves are in SAN notation.

    Handles full PGN move text including move numbers, comments in {}, variations
    in (), and result tokens. Only the main line is expanded.

    Parameters:
    -----------
    games_df : pd.DataFrame
        DataFrame containing chess games with FEN positions and move sequences in SAN notation
    fen_col : str
        Name of the column containing FEN positions (default: 'FEN')
    moves_col : str
        Name of the column containing move sequences in SAN notation (default: 'Moves')

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns 'fen' and 'move', one row per move in each game,
        with moves converted to UCI notation.
    """
    fens = []
    moves_out = []

    for moves_str in tqdm(games_df[moves_col]):
        visitor = _StrictGameBuilder()
        game = chess.pgn.read_game(io.StringIO(moves_str), Visitor=lambda: visitor)

        if game is None or visitor.has_error:
            continue

        board = game.board()
        for move in game.mainline_moves():
            fens.append(board.fen())
            moves_out.append(move.uci())
            board.push(move)

    return pd.DataFrame({"fen": fens, "move": moves_out})


def convert_san_to_uci(fen: str, san_move: str) -> str:
    """
    Convert a SAN chess move to UCI notation given the current board position.

    Parameters:
    -----------
    fen : str
        FEN string representing the current board position.
    san_move : str
        Move in Standard Algebraic Notation (e.g., 'Nf3', 'e4', 'O-O').

    Returns:
    --------
    str
        Move in UCI notation (e.g., 'g1f3', 'e2e4', 'e1g1').
    """
    board = chess.Board(fen)
    move = board.parse_san(san_move)
    return move.uci()


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

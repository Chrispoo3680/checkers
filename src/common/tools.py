import io
import logging
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union
from zipfile import ZipFile

import chess
import chess.pgn
import polars as pl
import torch
import yaml
from tqdm import tqdm

repo_root_dir: Path = Path(__file__).parent.parent.parent
sys.path.append(str(repo_root_dir))

PROMOTION_MAP = {None: 0, "n": 1, "b": 2, "r": 3, "q": 4}
INDEX_TO_PROMO = {v: k for k, v in PROMOTION_MAP.items()}

POLICY_PLANES = 73
POLICY_SIZE = 8 * 8 * POLICY_PLANES
# POLICY_SIZE = 20480

# AlphaZero-style policy planes:
# 0-55: sliding moves in 8 directions x 1..7 squares
# 56-63: knight moves (8)
# 64-72: underpromotions (N/B/R) x (left/forward/right)
_SLIDING_DIRS = [
    (0, 1),
    (1, 0),
    (0, -1),
    (-1, 0),
    (1, 1),
    (-1, 1),
    (1, -1),
    (-1, -1),
]
_KNIGHT_DIRS = [
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
]
_UNDERPROMO_PIECES = ["n", "b", "r"]


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

    terminal_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s:  %(message)s"
    )

    # Use tqdm-aware logging only for interactive terminals.
    if sys.stderr.isatty() and os.environ.get("TQDM_DISABLE", "0") != "1":
        tqdm_handler = TqdmLoggingHandler()
        tqdm_handler.setFormatter(terminal_formatter)
        logger.addHandler(tqdm_handler)
    else:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(terminal_formatter)
        logger.addHandler(stream_handler)

    # File handler for .log file output
    if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s:  %(message)s")
        )
        logger.addHandler(file_handler)

    return logger


def load_config():
    # Read in the configuration file
    with open(repo_root_dir / "config.yaml") as p:
        config = yaml.safe_load(p)
    return config


def get_temp_dir(create: bool = True) -> Path:
    config = load_config()
    temp_dir = repo_root_dir / config["tempfile_path"]

    if create:
        temp_dir.mkdir(parents=True, exist_ok=True)

    return temp_dir


def configure_temp_storage() -> Path:
    temp_dir = get_temp_dir(create=True).resolve()

    os.environ["TMPDIR"] = str(temp_dir)
    os.environ["TEMP"] = str(temp_dir)
    os.environ["TMP"] = str(temp_dir)
    tempfile.tempdir = str(temp_dir)

    return temp_dir


@contextmanager
def working_directory(path: Union[str, Path]):
    original_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original_cwd)


def rename_and_unzip_file(
    zip_file_path: Union[str, Path], new_file_path: Union[str, Path]
) -> None:
    with ZipFile(zip_file_path, "r") as zipped:
        zipped.extractall(path=new_file_path)

    os.remove(zip_file_path)


def get_files_from_folder(root_folder_path: Union[str, Path], extensions: list[str]):
    data_paths: list[Path] = []
    for root, _, files in os.walk(root_folder_path):
        if ".training" in root:
            continue  # Skip already processed parquet files
        for file_name in files:
            if file_name.endswith(tuple(extensions)):
                data_paths.append(Path(root) / file_name)
    return data_paths


def encode_fen_pos(fen: str) -> torch.Tensor:
    """
    Encode FEN into tensor (18, 8, 8)
    """

    board_fen, side, castling, ep_square = fen.split(" ")[:4]

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


def _uci_square_to_file_rank(square: str) -> tuple[int, int]:
    file_idx = ord(square[0]) - ord("a")
    rank_idx = int(square[1]) - 1
    return file_idx, rank_idx


def _file_rank_to_uci_square(file_idx: int, rank_idx: int) -> str:
    return chr(file_idx + ord("a")) + str(rank_idx + 1)


def encode_UCI_to_int(move_uci):

    if not isinstance(move_uci, str):
        raise ValueError(f"Invalid move type: {move_uci} ({type(move_uci)})")

    from_sq = square_to_index(move_uci[:2])
    from_file, from_rank = _uci_square_to_file_rank(move_uci[:2])
    to_file, to_rank = _uci_square_to_file_rank(move_uci[2:4])

    df = to_file - from_file
    dr = to_rank - from_rank

    # Underpromotions use dedicated planes.
    if len(move_uci) == 5 and move_uci[4] in _UNDERPROMO_PIECES:
        promo_piece = move_uci[4]
        forward_sign = 1 if dr > 0 else -1
        rel_df = df * forward_sign  # map to mover-relative left/forward/right
        if rel_df not in (-1, 0, 1):
            raise ValueError(f"Invalid underpromotion move: {move_uci}")
        piece_idx = _UNDERPROMO_PIECES.index(promo_piece)
        plane = 64 + piece_idx * 3 + (rel_df + 1)
        return from_sq * POLICY_PLANES + plane

    # Knight planes.
    if (df, dr) in _KNIGHT_DIRS:
        plane = 56 + _KNIGHT_DIRS.index((df, dr))
        return from_sq * POLICY_PLANES + plane

    # Sliding planes (includes queen promotions as normal forward moves).
    for dir_idx, (dir_df, dir_dr) in enumerate(_SLIDING_DIRS):
        if dir_df == 0:
            if df != 0:
                continue
            if dr == 0 or (dr > 0) != (dir_dr > 0):
                continue
            steps = abs(dr)
        elif dir_dr == 0:
            if dr != 0:
                continue
            if df == 0 or (df > 0) != (dir_df > 0):
                continue
            steps = abs(df)
        else:
            if abs(df) != abs(dr) or df == 0:
                continue
            if (df > 0) != (dir_df > 0) or (dr > 0) != (dir_dr > 0):
                continue
            steps = abs(df)

        if 1 <= steps <= 7:
            plane = dir_idx * 7 + (steps - 1)
            return from_sq * POLICY_PLANES + plane

    raise ValueError(f"Move cannot be represented in 8x8x73 policy space: {move_uci}")


def decode_int_to_UCI(index):
    from_sq = index // POLICY_PLANES
    plane = index % POLICY_PLANES

    from_square = index_to_square(from_sq)
    from_file, from_rank = _uci_square_to_file_rank(from_square)

    promotion = None
    if plane < 56:
        dir_idx = plane // 7
        steps = (plane % 7) + 1
        dir_df, dir_dr = _SLIDING_DIRS[dir_idx]
        to_file = from_file + dir_df * steps
        to_rank = from_rank + dir_dr * steps
    elif plane < 64:
        knight_idx = plane - 56
        dir_df, dir_dr = _KNIGHT_DIRS[knight_idx]
        to_file = from_file + dir_df
        to_rank = from_rank + dir_dr
    else:
        rel = (plane - 64) % 3 - 1  # -1,0,1
        piece_idx = (plane - 64) // 3
        promotion = _UNDERPROMO_PIECES[piece_idx]

        forward_sign = 1 if from_rank >= 6 else -1
        df = rel * forward_sign
        dr = forward_sign
        to_file = from_file + df
        to_rank = from_rank + dr

    if not (0 <= to_file < 8 and 0 <= to_rank < 8):
        raise ValueError(f"Decoded move goes out of board: index={index}")

    to_square = _file_rank_to_uci_square(to_file, to_rank)
    if promotion is None:
        return from_square + to_square
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

    for row in tqdm(puzzles_df.iter_rows(named=True), total=len(puzzles_df)):
        board = chess.Board(row[fen_col])

        # Split moves if it's a string, otherwise assume it's already a list
        if isinstance(row[moves_col], str):
            moves = row[moves_col].split()
        else:
            moves = row[moves_col]

        for move_uci in moves:
            # Store current position and move
            expanded_data.append({"fen": board.fen(), "moves": move_uci})

            # Make the move to get to next position
            try:
                move = chess.Move.from_uci(move_uci)
                board.push(move)
            except ValueError:
                # Skip invalid moves
                continue

    return expanded_data


def _StrictGameBuilder():
    """GameBuilder that flags games containing illegal moves."""

    class Builder(chess.pgn.GameBuilder):
        def __init__(self):
            super().__init__()
            self.has_error = False

        def handle_error(self, error):
            self.has_error = True

    return Builder()


def expand_game_positions_san(games_df, moves_col="Moves", eval_col="Evaluation"):
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
    evaluations = []

    for moves_str, winner in tqdm(
        zip(games_df[moves_col], games_df[eval_col]), total=len(games_df)
    ):
        visitor = _StrictGameBuilder()
        game = chess.pgn.read_game(io.StringIO(moves_str), Visitor=lambda: visitor)

        if game is None or visitor.has_error:
            continue

        board = game.board()
        for i, move in enumerate(game.mainline_moves()):
            fens.append(board.fen())
            moves_out.append(move.uci())
            evaluations.append(
                winner * ((-1) ** (i % 2))
            )  # +1 for white win, -1 for black win, 0 for draw
            board.push(move)

    return {"fen": fens, "moves": moves_out, "evaluation": evaluations}


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


def cp_weighted_moves(cp_values: list[int], temperature=75):
    best_cp = cp_values[0]
    delta_cps = [best_cp - cp for cp in cp_values]
    weights = [pow(2.71828, -delta_cp / temperature) for delta_cp in delta_cps]
    total_weight = sum(weights)
    normalized_weights = [weight / total_weight for weight in weights]
    return normalized_weights


def mate_to_cp(mate_expr, cp_expr):
    """Use mate score as large cp sentinel, fall back to cp if no mate."""
    return (
        pl.when(mate_expr.is_not_null())
        .then(
            pl.when(mate_expr > 0)
            .then(30000 - mate_expr * 10)  # mate=1 → 29990, mate=3 → 29970
            .otherwise(-30000 - mate_expr * 10)  # mate=-1 → -29990, mate=-3 → -29970
        )
        .otherwise(cp_expr)
    )


def effective_cp_element():
    """For use inside list.agg (pl.element() context)."""
    return mate_to_cp(
        pl.element().struct.field("mate"),
        pl.element().struct.field("cp"),
    )


def effective_cp_single(pvs_expr):
    """For use on a single struct (e.g. after list.get(0))."""
    return mate_to_cp(
        pvs_expr.struct.field("mate"),
        pvs_expr.struct.field("cp"),
    )


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

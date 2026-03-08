"""Generate a Stockfish-annotated chess dataset.

For each sampled FEN position this script records:
  - fen          : chess position in FEN notation
  - move_1..5    : top 5 best moves in UCI notation
  - evaluation   : evaluation in centipawns from the side-to-move's perspective
                   (positive = good for the side to move).
                   Forced mates are stored as "M<n>", where a positive n means
                   the side to move wins in n moves and a negative n means the
                   side to move loses in |n| moves.

Usage example:
    python generate_stockfish_dataset.py \\
        --stockfish /usr/bin/stockfish \\
        --depth 15 \\
        --workers 8 \\
        --sample 500000 \\
        --output data/stockfish_dataset/stockfish_dataset.csv
"""

import argparse
import csv
import multiprocessing as mp
import os
import sys
from pathlib import Path

import chess
import chess.engine
import pandas as pd
from tqdm import tqdm

repo_root_dir: Path = Path(__file__).parent
sys.path.append(str(repo_root_dir))

from src.common import tools

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_TOP_MOVES = 5

# ---------------------------------------------------------------------------
# Per-process globals (one Stockfish instance per worker)
# ---------------------------------------------------------------------------

_engine: chess.engine.SimpleEngine | None = None
_stockfish_path: str = ""
_depth: int = 15
_time_limit: float | None = None
_hash_mb: int = 16


def _worker_init(
    stockfish_path: str,
    depth: int,
    time_limit: float | None,
    hash_mb: int,
) -> None:
    """Initialise a Stockfish engine instance inside each worker process."""
    global _engine, _stockfish_path, _depth, _time_limit, _hash_mb
    _stockfish_path = stockfish_path
    _depth = depth
    _time_limit = time_limit
    _hash_mb = hash_mb
    _engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    _engine.configure({"Hash": hash_mb, "Threads": 1})


def _get_limit() -> chess.engine.Limit:
    if _time_limit is not None:
        return chess.engine.Limit(time=_time_limit)
    return chess.engine.Limit(depth=_depth)


def _restart_engine() -> bool:
    """Try to restart the engine after a crash; returns True on success."""
    global _engine
    try:
        if _engine is not None:
            _engine.quit()
    except Exception:
        pass
    try:
        _engine = chess.engine.SimpleEngine.popen_uci(_stockfish_path)
        _engine.configure({"Hash": _hash_mb, "Threads": 1})
        return True
    except Exception:
        return False


def _analyse(fen: str) -> tuple | None:
    """Analyse one FEN position; returns a flat tuple or None on failure."""
    global _engine

    # Validate FEN
    try:
        board = chess.Board(fen)
    except ValueError:
        return None

    # Skip positions that are already over
    if board.is_game_over():
        return None

    # Run Stockfish analysis, restarting once on engine error
    for attempt in range(2):
        try:
            info_list: list[chess.engine.InfoDict] = _engine.analyse(
                board,
                _get_limit(),
                multipv=NUM_TOP_MOVES,
            )
            break
        except chess.engine.EngineError:
            if attempt == 0 and _restart_engine():
                continue
            return None
        except Exception:
            return None

    moves: list[str | None] = []
    evaluation: int | str | None = None

    for idx, info in enumerate(info_list):
        pv = info.get("pv")
        moves.append(pv[0].uci() if pv else None)

        if idx == 0:
            # Score from the perspective of the side to move
            score = info["score"].pov(board.turn)
            if score.is_mate():
                evaluation = f"M{score.mate()}"
            else:
                evaluation = score.score()

    # Pad to NUM_TOP_MOVES in case Stockfish returned fewer lines
    while len(moves) < NUM_TOP_MOVES:
        moves.append(None)

    return (fen, *moves[:NUM_TOP_MOVES], evaluation)


# Module-level wrapper required for multiprocessing pickling
def _work(fen: str) -> tuple | None:
    return _analyse(fen)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Stockfish-annotated chess position dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stockfish",
        type=str,
        default="stockfish",
        help="Path to the Stockfish executable ('stockfish' assumes it is on PATH).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=15,
        help="Search depth for Stockfish analysis. Ignored when --time is set.",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=None,
        dest="time_limit",
        help="Time budget in seconds per position. When set, overrides --depth.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=500_000,
        help="Number of positions to sample from the source dataset.",
    )
    # Use physical core count as default to avoid SMT contention
    # (chess engines are CPU-bound and don't benefit from hyperthreading siblings)
    try:
        with open("/proc/cpuinfo") as f:
            physical_cores = (
                len(
                    {
                        line.split(":")[1].strip()
                        for line in f
                        if line.startswith("core id")
                    }
                )
                or 1
            )
    except Exception:
        physical_cores = max(1, (os.cpu_count() or 2) // 2)

    parser.add_argument(
        "--workers",
        type=int,
        default=physical_cores,
        help="Number of parallel Stockfish worker processes. Default uses physical core count to avoid SMT contention.",
    )
    parser.add_argument(
        "--hash",
        type=int,
        default=16,
        dest="hash_mb",
        help="Hash table size in MB per Stockfish worker. Total RAM used = workers × hash.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/chess_positions/positions.csv",
        help="Source CSV file containing a 'fen' column.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/stockfish_dataset/stockfish_dataset.csv",
        help="Destination CSV path for the generated dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1,
        help="Number of positions dispatched to each worker per batch.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    config = tools.load_config()
    log_dir = Path(config.get("logging_path", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = tools.create_logger(
        logger_name=__name__,
        log_path=log_dir / "generate_stockfish_dataset.log",
    )

    # ── Verify Stockfish is reachable ────────────────────────────────────────
    try:
        test_engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
        test_engine.quit()
        logger.info(f"Stockfish found at: {args.stockfish!r}")
    except FileNotFoundError:
        logger.error(
            f"Stockfish executable not found at {args.stockfish!r}. "
            "Install Stockfish and pass the correct path via --stockfish."
        )
        sys.exit(1)

    # ── Load and sample source positions ────────────────────────────────────
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    logger.info(f"Loading positions from {input_path} …")
    positions_df = pd.read_csv(input_path, usecols=["fen"])
    logger.info(f"Total positions available: {len(positions_df):,}")

    sample_size = min(args.sample, len(positions_df))
    sampled_fens: list[str] = (
        positions_df["fen"].sample(n=sample_size, random_state=args.seed).tolist()
    )
    logger.info(f"Sampled {sample_size:,} positions (seed={args.seed})")

    # ── Resume: skip already-processed FENs ──────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    already_done: set[str] = set()
    if output_path.exists():
        try:
            existing_df = pd.read_csv(output_path, usecols=["fen"])
            already_done = set(existing_df["fen"].tolist())
            logger.info(
                f"Resuming: {len(already_done):,} positions already in output file."
            )
        except Exception:
            logger.warning("Could not read existing output file; starting fresh.")

    todo_fens = [f for f in sampled_fens if f not in already_done]
    logger.info(f"Positions left to analyse: {len(todo_fens):,}")

    if not todo_fens:
        logger.info("Nothing to do — output is already complete.")
        return

    # ── Analysis ─────────────────────────────────────────────────────────────
    columns = (
        ["fen"] + [f"move_{i}" for i in range(1, NUM_TOP_MOVES + 1)] + ["evaluation"]
    )
    write_header = not output_path.exists() or output_path.stat().st_size == 0

    limit_desc = (
        f"time={args.time_limit}s"
        if args.time_limit is not None
        else f"depth={args.depth}"
    )
    logger.info(
        f"Starting analysis — {limit_desc}, workers={args.workers}, "
        f"chunksize={args.chunksize}"
    )

    processed = 0
    failed = 0
    flush_interval = 5_000

    # Use 'spawn' context to avoid asyncio/epoll state being inherited by
    # forked workers (chess.engine uses asyncio internally, which breaks with fork)
    spawn_ctx = mp.get_context("spawn")

    try:
        with open(output_path, "a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            if write_header:
                writer.writerow(columns)

            with spawn_ctx.Pool(
                processes=args.workers,
                initializer=_worker_init,
                initargs=(args.stockfish, args.depth, args.time_limit, args.hash_mb),
            ) as pool:
                with tqdm(
                    total=len(todo_fens),
                    unit="pos",
                    desc="Analysing",
                ) as pbar:
                    for result in pool.imap_unordered(
                        _work, todo_fens, chunksize=args.chunksize
                    ):
                        if result is not None:
                            writer.writerow(result)
                            processed += 1
                        else:
                            failed += 1
                        pbar.update(1)

                        if (processed + failed) % flush_interval == 0:
                            csv_file.flush()

    except KeyboardInterrupt:
        logger.warning("Interrupted by user — partial results saved to output file.")

    logger.info(
        f"Done.  Processed: {processed:,}  |  Failed/skipped: {failed:,}  |  "
        f"Output: {output_path}"
    )


if __name__ == "__main__":
    main()

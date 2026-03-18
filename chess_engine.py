#!/usr/bin/env python3
"""UCI-compatible wrapper for the Checkers chess model.

This engine loads a Checkers checkpoint and serves moves over the Universal
Chess Interface (UCI) protocol so it can be used by GUI/tournament tools
such as cutechess-cli.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chess
import torch

from src.common import tools
from src.evaluate.mcts import MCTS
from src.model.models import CheckersNetV3

ENGINE_NAME = "Checkers-UCI"
ENGINE_AUTHOR = "Christoffer Brandt"
DEFAULT_MODEL_NAME = "checkers4.2"
DEFAULT_NUM_SIMULATIONS = 1600
DEFAULT_C_PUCT = 2.5

CONFIG = tools.load_config()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Checkers model as a UCI-compatible chess engine."
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Checkpoint name in models/, without .pt suffix.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional explicit checkpoint path. Overrides --model-name.",
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=DEFAULT_NUM_SIMULATIONS,
        help="Default MCTS simulations per move.",
    )
    parser.add_argument(
        "--c-puct",
        type=float,
        default=DEFAULT_C_PUCT,
        help="MCTS exploration constant.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device selection for inference.",
    )
    return parser.parse_args()


def resolve_checkpoint_path(model_name: str, explicit_checkpoint: Path | None) -> Path:
    if explicit_checkpoint is not None:
        return explicit_checkpoint
    return Path(__file__).resolve().parent / CONFIG["model_path"] / f"{model_name}.pt"


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: Path, device: torch.device) -> CheckersNetV3:
    model = CheckersNetV3(
        input_planes=18,
        policy_planes=73,
        channels=256,
        num_blocks=15,
        se_every_n_blocks=4,
        temperature=1.2,
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


class UCICheckersEngine:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        model_name: str,
        default_sims: int,
        default_c_puct: float,
    ):
        self.model = model
        self.device = device
        self.model_name = model_name
        self.board = chess.Board()

        self.num_simulations = max(1, int(default_sims))
        self.c_puct = float(default_c_puct)

    @staticmethod
    def _send(line: str) -> None:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    def _make_mcts(self) -> MCTS:
        return MCTS(
            model=self.model,
            num_simulations=self.num_simulations,
            c_puct=self.c_puct,
            device=str(self.device),
        )

    @staticmethod
    def _parse_go_limits(tokens: list[str]) -> dict[str, int | bool]:
        """Parse standard UCI `go` limits into a dict.

        Supported keys: wtime, btime, winc, binc, movestogo, movetime,
        depth, nodes, mate, infinite.
        """
        limits: dict[str, int | bool] = {"infinite": False}

        i = 1
        while i < len(tokens):
            key = tokens[i]
            if key == "infinite":
                limits["infinite"] = True
                i += 1
                continue

            if key in {
                "wtime",
                "btime",
                "winc",
                "binc",
                "movestogo",
                "movetime",
                "depth",
                "nodes",
                "mate",
            }:
                if i + 1 < len(tokens):
                    try:
                        limits[key] = int(tokens[i + 1])
                    except ValueError:
                        pass
                i += 2
                continue

            i += 1

        return limits

    def _sims_for_go(self, limits: dict[str, int | bool]) -> int:
        """Compute a conservative simulation budget from UCI time controls."""
        base = max(1, self.num_simulations)

        # Explicit movetime has priority.
        movetime = limits.get("movetime")
        if isinstance(movetime, int) and movetime > 0:
            if movetime < 100:
                return max(1, base // 32)
            if movetime < 250:
                return max(1, base // 16)
            if movetime < 500:
                return max(1, base // 8)
            if movetime < 1000:
                return max(1, base // 4)
            if movetime < 2000:
                return max(1, base // 2)
            return base

        # Depth/nodes limits are mapped coarsely to lower compute.
        depth = limits.get("depth")
        if isinstance(depth, int) and depth > 0:
            if depth <= 2:
                return max(1, base // 32)
            if depth <= 4:
                return max(1, base // 16)
            if depth <= 6:
                return max(1, base // 8)
            if depth <= 8:
                return max(1, base // 4)

        # If a clock is provided, reserve plenty of time buffer.
        side_time_key = "wtime" if self.board.turn == chess.WHITE else "btime"
        side_inc_key = "winc" if self.board.turn == chess.WHITE else "binc"

        remaining_ms = limits.get(side_time_key)
        increment_ms = limits.get(side_inc_key)
        moves_to_go = limits.get("movestogo")

        if isinstance(remaining_ms, int) and remaining_ms > 0:
            mtg = (
                moves_to_go if isinstance(moves_to_go, int) and moves_to_go > 0 else 30
            )
            inc = (
                increment_ms
                if isinstance(increment_ms, int) and increment_ms > 0
                else 0
            )

            target_ms = max(25, remaining_ms // max(10, mtg) + inc // 2)

            if target_ms < 100:
                return max(1, base // 32)
            if target_ms < 250:
                return max(1, base // 16)
            if target_ms < 500:
                return max(1, base // 8)
            if target_ms < 1000:
                return max(1, base // 4)
            if target_ms < 2000:
                return max(1, base // 2)
            return base

        return base

    def _handle_uci(self) -> None:
        self._send(f"id name {ENGINE_NAME} ({self.model_name})")
        self._send(f"id author {ENGINE_AUTHOR}")
        self._send(
            f"option name NumSimulations type spin default {self.num_simulations} min 1 max 100000"
        )
        self._send(f"option name CPuct type string default {self.c_puct}")
        self._send("uciok")

    def _handle_setoption(self, tokens: list[str]) -> None:
        # UCI format: setoption name <name> [value <value>]
        if len(tokens) < 3 or tokens[1] != "name":
            return

        value_index = None
        for idx, token in enumerate(tokens):
            if token == "value":
                value_index = idx
                break

        if value_index is None:
            name_tokens = tokens[2:]
            value_tokens: list[str] = []
        else:
            name_tokens = tokens[2:value_index]
            value_tokens = tokens[value_index + 1 :]

        option_name = " ".join(name_tokens).strip().lower()
        option_value = " ".join(value_tokens).strip()

        if option_name == "numsimulations":
            try:
                self.num_simulations = max(1, int(option_value))
            except ValueError:
                return
            return

        if option_name == "cpuct":
            try:
                self.c_puct = float(option_value)
            except ValueError:
                return

    def _handle_position(self, tokens: list[str]) -> None:
        # UCI format:
        # position startpos [moves m1 m2 ...]
        # position fen <fen> [moves m1 m2 ...]
        if len(tokens) < 2:
            return

        idx = 1
        if tokens[idx] == "startpos":
            self.board = chess.Board()
            idx += 1
        elif tokens[idx] == "fen":
            fen_fields = tokens[idx + 1 : idx + 7]
            if len(fen_fields) < 6:
                return
            fen = " ".join(fen_fields)
            try:
                self.board = chess.Board(fen)
            except ValueError:
                return
            idx += 7
        else:
            return

        if idx < len(tokens) and tokens[idx] == "moves":
            idx += 1
            for move_uci in tokens[idx:]:
                try:
                    move = chess.Move.from_uci(move_uci)
                except ValueError:
                    break
                if move in self.board.legal_moves:
                    self.board.push(move)
                else:
                    break

    @torch.no_grad()
    def _choose_bestmove(self, simulations: int | None = None) -> chess.Move | None:
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return None

        if simulations is None:
            simulations = self.num_simulations

        mcts = MCTS(
            model=self.model,
            num_simulations=max(1, int(simulations)),
            c_puct=self.c_puct,
            device=str(self.device),
        )
        visits = mcts.run(self.board)

        if not visits:
            return legal_moves[0]

        return max(visits, key=visits.get)

    def _handle_go(self, tokens: list[str]) -> None:
        limits = self._parse_go_limits(tokens)
        simulations = self._sims_for_go(limits)

        best_move = self._choose_bestmove(simulations=simulations)
        if best_move is None:
            self._send("bestmove 0000")
            return
        self._send(f"bestmove {best_move.uci()}")

    def run(self) -> None:
        while True:
            raw = sys.stdin.readline()
            if raw == "":
                break

            line = raw.strip()
            if not line:
                continue

            tokens = line.split()
            cmd = tokens[0]

            if cmd == "uci":
                self._handle_uci()
            elif cmd == "isready":
                self._send("readyok")
            elif cmd == "ucinewgame":
                self.board = chess.Board()
            elif cmd == "position":
                self._handle_position(tokens)
            elif cmd == "go":
                self._handle_go(tokens)
            elif cmd == "setoption":
                self._handle_setoption(tokens)
            elif cmd == "stop":
                # Search is synchronous; no background task to stop.
                continue
            elif cmd == "ponderhit":
                continue
            elif cmd == "quit":
                break


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    checkpoint_path = resolve_checkpoint_path(args.model_name, args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = load_model(checkpoint_path, device)

    engine = UCICheckersEngine(
        model=model,
        device=device,
        model_name=args.model_name,
        default_sims=args.num_simulations,
        default_c_puct=args.c_puct,
    )
    engine.run()


if __name__ == "__main__":
    main()

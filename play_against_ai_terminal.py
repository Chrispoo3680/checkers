#!/usr/bin/env python3
"""Play against the trained chess model in a Linux terminal.

This script mirrors the gameplay features from notebooks/model_predictions.ipynb,
but uses typed UCI moves instead of mouse input.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import chess
import torch

from src.evaluate.mcts import MCTS
from src.model.models import CheckersNetV3
from src.common import tools


DEFAULT_MODEL_NAME = "checkers4.2"
DEFAULT_NUM_SIMULATIONS = 400
DEFAULT_C_PUCT = 2.5

CONFIG = tools.load_config()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play a terminal chess game against the CheckersNet model using UCI moves."
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
        help="MCTS simulations per AI move.",
    )
    parser.add_argument(
        "--c-puct",
        type=float,
        default=DEFAULT_C_PUCT,
        help="MCTS exploration constant.",
    )
    parser.add_argument(
        "--human-color",
        choices=["white", "black"],
        default="white",
        help="Which side you play.",
    )
    return parser.parse_args()


def get_checkpoint_path(model_name: str, explicit_checkpoint: Path | None) -> Path:
    if explicit_checkpoint is not None:
        return explicit_checkpoint
    return Path(__file__).resolve().parent / CONFIG["model_path"] / f"{model_name}.pt"


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


def board_to_terminal(board: chess.Board, human_color: chess.Color) -> str:
    if human_color == chess.WHITE:
        rank_order = range(7, -1, -1)
        file_order = range(0, 8)
        file_labels = "a b c d e f g h"
    else:
        # Rotate board 180 degrees so Black is shown at the bottom.
        rank_order = range(0, 8)
        file_order = range(7, -1, -1)
        file_labels = "h g f e d c b a"

    lines: list[str] = []
    lines.append("    " + file_labels)
    lines.append("  +-----------------+")

    for rank in rank_order:
        row_cells: list[str] = []
        for file_ in file_order:
            square = chess.square(file_, rank)
            piece = board.piece_at(square)
            row_cells.append(piece.symbol() if piece else ".")
        lines.append(f"{rank + 1} | {' '.join(row_cells)} |")

    lines.append("  +-----------------+")
    return "\n".join(lines)


def format_move_log(board: chess.Board) -> str:
    moves = list(board.move_stack)
    if not moves:
        return "(no moves yet)"

    replay = chess.Board()
    move_texts: list[str] = []
    for move in moves:
        move_texts.append(replay.san(move))
        replay.push(move)

    lines: list[str] = []
    for i in range(0, len(move_texts), 2):
        move_num = i // 2 + 1
        white_move = move_texts[i]
        black_move = move_texts[i + 1] if i + 1 < len(move_texts) else ""
        lines.append(f"{move_num}. {white_move} {black_move}".rstrip())

    return "\n".join(lines)


def game_status(board: chess.Board, ai_to_move: bool) -> tuple[bool, str]:
    if board.is_checkmate():
        return True, "Checkmate! AI wins!" if ai_to_move else "Checkmate! You win!"
    if board.is_stalemate():
        return True, "Stalemate! Draw."
    if board.is_insufficient_material():
        return True, "Draw - insufficient material."
    if board.can_claim_threefold_repetition():
        return True, "Draw - threefold repetition."
    return False, ""


@torch.no_grad()
def get_model_move(
    board: chess.Board,
    model: torch.nn.Module,
    mcts: MCTS,
    device: torch.device,
) -> tuple[chess.Move | None, float, float]:
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None, 0.0, 0.0

    model.eval()
    visit_counts = mcts.run(board)

    if not visit_counts:
        return None, 0.0, 0.0

    best_move = max(visit_counts, key=visit_counts.get)
    total_visits = sum(visit_counts.values())
    visit_share = visit_counts[best_move] / total_visits if total_visits > 0 else 0.0

    from src.common import tools

    position_tensor = tools.encode_fen_pos(board.fen()).unsqueeze(0).to(device)
    _, value_tensor = model(position_tensor)
    value = value_tensor.item()

    return best_move, visit_share, value


def print_help() -> None:
    print("Commands:")
    print("  <uci>  Play a move in UCI, e.g. e2e4, g1f3, e7e8q")
    print("  z      Undo (removes AI + human move when possible)")
    print("  r      Reset game")
    print("  s      Switch side (and reset game)")
    print("  q      Quit")
    print("  h      Show this help")


def print_state(
    board: chess.Board,
    human_color: chess.Color,
    model_name: str,
    ai_move_str: str | None,
    ai_visit_share: float | None,
    ai_value: float | None,
    status_msg: str,
) -> None:
    print("\n" + "=" * 72)
    print(f"Model: {model_name}")
    print(board_to_terminal(board, human_color))

    turn_text = "White to move" if board.turn == chess.WHITE else "Black to move"
    you_text = "White" if human_color == chess.WHITE else "Black"
    print(f"Turn: {turn_text} | Move: {board.fullmove_number} | You: {you_text}")

    print("\nAI Analysis")
    print(f"  Last AI move: {ai_move_str if ai_move_str else '-'}")
    print(
        f"  Visit share: {ai_visit_share:.4f}"
        if ai_visit_share is not None
        else "  Visit share: -"
    )
    if ai_value is None:
        print("  Position val: -")
        print("  Eval (cp): -")
    else:
        print(f"  Position val: {ai_value:+.3f}")
        print(f"  Eval (cp): {ai_value * 1000:+.0f}")

    print("\nMove Log")
    print(format_move_log(board))

    if status_msg:
        print(f"\nStatus: {status_msg}")

    print("\nType 'h' for commands.")
    print("=" * 72)


def parse_uci_move(move_text: str) -> chess.Move | None:
    try:
        return chess.Move.from_uci(move_text)
    except ValueError:
        return None


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = get_checkpoint_path(args.model_name, args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = load_model(checkpoint_path, device)
    mcts = MCTS(
        model=model,
        num_simulations=args.num_simulations,
        c_puct=args.c_puct,
        device=str(device),
    )

    human_color = chess.WHITE if args.human_color == "white" else chess.BLACK
    ai_color = chess.BLACK if human_color == chess.WHITE else chess.WHITE

    board = chess.Board()
    game_over = False
    status_msg = ""
    ai_visit_share: float | None = None
    ai_value: float | None = None
    ai_move_str: str | None = None

    print(f"Using device: {device}")
    print(f"Loaded checkpoint: {checkpoint_path}")
    print_help()

    while True:
        print_state(
            board=board,
            human_color=human_color,
            model_name=args.model_name,
            ai_move_str=ai_move_str,
            ai_visit_share=ai_visit_share,
            ai_value=ai_value,
            status_msg=status_msg,
        )

        ai_turn = board.turn == ai_color and not game_over
        if ai_turn:
            print(f"AI thinking... ({args.num_simulations} sims)")
            best_move, visit_share, value = get_model_move(board, model, mcts, device)

            if best_move and best_move in board.legal_moves:
                ai_move_str = board.san(best_move)
                board.push(best_move)
                ai_visit_share = visit_share
                ai_value = value
                print(f"AI plays: {ai_move_str} ({best_move.uci()})")
                print(
                    f"Position evaluation: {ai_value * 1000:.1f} cp | visit share: {ai_visit_share:.4f}"
                )
            else:
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    fallback = random.choice(legal_moves)
                    ai_move_str = board.san(fallback)
                    board.push(fallback)
                    ai_visit_share = 0.0
                    ai_value = value if best_move is not None else 0.0
                    print("AI fallback used (random legal move).")
                    print(f"AI plays: {ai_move_str} ({fallback.uci()})")

            game_over, status_msg = game_status(board, ai_to_move=False)
            continue

        user_input = input("Your move (UCI or command): ").strip().lower()

        if user_input in {"q", "quit", "exit"}:
            print("Goodbye.")
            return

        if user_input in {"h", "help"}:
            print_help()
            continue

        if user_input in {"r", "reset"}:
            board.reset()
            game_over = False
            status_msg = ""
            ai_visit_share = None
            ai_value = None
            ai_move_str = None
            print("Game reset.")
            continue

        if user_input in {"s", "switch"}:
            human_color = chess.BLACK if human_color == chess.WHITE else chess.WHITE
            ai_color = chess.BLACK if ai_color == chess.WHITE else chess.WHITE
            board.reset()
            game_over = False
            status_msg = ""
            ai_visit_share = None
            ai_value = None
            ai_move_str = None
            side = "White" if human_color == chess.WHITE else "Black"
            print(f"Switched side. You now play {side}.")
            continue

        if user_input in {"z", "undo"}:
            if len(board.move_stack) >= 2:
                board.pop()
                board.pop()
                print("Undid the last two half-moves (AI + human).")
            elif len(board.move_stack) == 1:
                board.pop()
                print("Undid the last half-move.")
            else:
                print("No moves to undo.")

            game_over = False
            status_msg = ""
            continue

        if game_over:
            print("Game is over. Use 'r' to reset, 's' to switch side, or 'q' to quit.")
            continue

        move = parse_uci_move(user_input)
        if move is None:
            print("Invalid UCI move format. Example: e2e4, g1f3, e7e8q")
            continue

        if move not in board.legal_moves:
            print("Illegal move in current position.")
            continue

        board.push(move)
        game_over, status_msg = game_status(board, ai_to_move=True)


if __name__ == "__main__":
    main()

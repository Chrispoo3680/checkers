"""
AlphaZero-style Monte Carlo Tree Search for chess.

Uses a dual-head neural network (policy + value) and the python-chess
library for board state management.
"""

import math
import time
from typing import Dict, List, Optional

import chess
import torch
import torch.nn.functional as F

from src.common.tools import POLICY_SIZE, encode_fen_pos, encode_UCI_to_int


class Node:
    """A single node in the MCTS tree.

    Each node corresponds to a board state reached by playing ``move``
    from the parent node.  It stores visit statistics used by the PUCT
    selection formula and holds references to its children (one per
    legal move explored so far).

    Attributes:
        parent: Parent node, or ``None`` for the root.
        children: Dict mapping ``chess.Move`` to child ``Node``.
        move: The move that led to this node from its parent.
        prior: Prior probability assigned by the policy network.
        visit_count: Number of times this node has been visited.
        value_sum: Cumulative backed-up value (from this node's perspective).
    """

    def __init__(
        self,
        parent: Optional["Node"] = None,
        move: Optional[chess.Move] = None,
        prior: float = 0.0,
    ):
        self.parent = parent
        self.move = move
        self.prior = prior

        self.children: Dict[chess.Move, "Node"] = {}
        self.visit_count: int = 0
        self.value_sum: float = 0.0

    def expanded(self) -> bool:
        """Return ``True`` if this node has been expanded (has children)."""
        return len(self.children) > 0

    def value(self) -> float:
        """Return the mean action-value Q(s, a).

        Returns 0.0 for unvisited nodes to avoid division by zero.
        """
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expand(self, policy: Dict[chess.Move, float]) -> None:
        """Create child nodes for every legal move using the given policy.

        Args:
            policy: Dict mapping each legal move to its prior probability
                    (already normalised over legal moves).
        """
        for move, prob in policy.items():
            self.children[move] = Node(parent=self, move=move, prior=prob)

    def select_child(self, c_puct: float) -> tuple:
        """Select the child with the highest PUCT score.

        The Upper Confidence bound applied to Trees (PUCT) formula used
        by AlphaZero is::

            PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))

        where *Q* is the mean value, *P* the prior, *N(s)* the parent
        visit count, and *N(s, a)* the child visit count.

        Args:
            c_puct: Exploration constant controlling the prior's influence.

        Returns:
            A tuple ``(best_move, best_child)`` with the best action and
            the corresponding child node.
        """
        best_score = -float("inf")
        best_move = None
        best_child = None

        sqrt_parent = math.sqrt(self.visit_count)

        for move, child in self.children.items():
            # Negate child's Q because child stores value from the child's
            # perspective (the opponent), but the parent needs it from its own
            # perspective to maximise correctly.
            q = -child.value()
            u = c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            score = q + u

            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        return best_move, best_child


class MCTS:
    """AlphaZero-style Monte Carlo Tree Search.

    Uses a neural network to evaluate leaf positions and to provide
    move priors, then runs *num_simulations* playouts from the given
    root position to build a search tree.

    Args:
        model: A PyTorch model whose ``forward(x)`` returns
               ``(policy_logits, value)`` where policy logits are in the
               8x8x73 space (shape ``(1, 8, 8, 73)`` or flatten-equivalent)
               and ``(1, 1)`` respectively.
        num_simulations: How many MCTS simulations to run per search call.
        c_puct: Exploration constant for the PUCT formula.
        device: Torch device for inference (default: ``"cpu"``).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        num_simulations: int,
        c_puct: float,
        device: str = "cpu",
    ):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = torch.device(device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run(
        self,
        root_board: chess.Board,
        max_time_s: float | None = None,
        min_simulations: int = 1,
    ) -> Dict[chess.Move, int]:
        """Run MCTS from *root_board* and return visit counts.

        Args:
            root_board: The current board position.  **Not** mutated.
            max_time_s: Optional wall-clock budget in seconds. Search stops
                when this deadline is reached, after at least
                ``min_simulations`` playouts.
            min_simulations: Minimum number of playouts to execute before the
                time limit may terminate search.

        Returns:
            A dict mapping each legal move to the number of times it was
            visited during the search.
        """
        root = Node()

        # Expand the root using the network policy.
        policy, value = self._evaluate(root_board)
        root.expand(policy)
        root.visit_count = 1
        root.value_sum = value  # seed with initial network evaluation

        min_simulations = max(0, int(min_simulations))
        deadline = None
        if max_time_s is not None and max_time_s > 0:
            deadline = time.perf_counter() + float(max_time_s)

        for sim_idx in range(self.num_simulations):
            if (
                deadline is not None
                and sim_idx >= min_simulations
                and time.perf_counter() >= deadline
            ):
                break

            node = root
            board = root_board.copy()
            search_path: List[Node] = [node]

            # --- Selection ---
            while node.expanded() and not board.is_game_over():
                move, node = node.select_child(self.c_puct)
                board.push(move)
                search_path.append(node)

            # --- Evaluate / Expand ---
            value = self._get_leaf_value(node, board)

            # --- Back-propagation ---
            self._backpropagate(search_path, value)

        return {move: child.visit_count for move, child in root.children.items()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_leaf_value(self, node: Node, board: chess.Board) -> float:
        """Evaluate a leaf node, expanding it if the game is not over.

        If the position is terminal the value is determined by the game
        outcome.  Otherwise the neural network is used to both evaluate
        the position and provide priors for expansion.

        Args:
            node: The leaf node reached during selection.
            board: Board state corresponding to *node*.

        Returns:
            The position value from the perspective of the player
            **to move** at this node.
        """
        outcome = board.outcome()
        if outcome is not None:
            if outcome.winner is None:
                return 0.0
            # +1 if the side to move has won (shouldn't normally happen —
            # the previous player delivered mate), -1 otherwise.
            side_to_move = board.turn  # True = White
            return 1.0 if outcome.winner == side_to_move else -1.0

        # Non-terminal leaf — expand with network policy & return value.
        policy, value = self._evaluate(board)
        node.expand(policy)
        return value

    def _evaluate(self, board: chess.Board) -> tuple:
        """Run the neural network on *board*.

        Returns:
            A tuple ``(policy, value)`` where *policy* is a dict mapping
            each legal move to its normalised prior probability and *value*
            is a float in [-1, 1] from the perspective of the side to move.
        """
        board_tensor = encode_fen_pos(board.fen()).unsqueeze(0).to(self.device)
        policy_logits, value = self.model(board_tensor)

        # Flatten to 1-D (model may return (1, POLICY_SIZE)).
        policy_logits = policy_logits.view(-1)
        value = value.item()

        # --- Mask illegal moves ---
        legal_moves = list(board.legal_moves)
        legal_indices = [encode_UCI_to_int(move.uci()) for move in legal_moves]

        mask = torch.full((POLICY_SIZE,), float("-inf"), device=policy_logits.device)
        mask[legal_indices] = 0.0
        masked_logits = policy_logits + mask

        # Softmax over legal moves only.
        probs = F.softmax(masked_logits, dim=0)

        policy = {
            move: probs[idx].item() for move, idx in zip(legal_moves, legal_indices)
        }

        return policy, value

    @staticmethod
    def _backpropagate(search_path: List[Node], value: float) -> None:
        """Back-propagate *value* up the search path.

        The value alternates sign at each level because consecutive
        nodes represent opposing players.

        Args:
            search_path: List of nodes from root to the evaluated leaf.
            value: Evaluation from the perspective of the player to move
                   at the **leaf** node.
        """
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value  # flip perspective

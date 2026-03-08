import chess
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..common import tools, utils


class ChessDataset(Dataset):
    def __init__(self, data_df: pd.DataFrame):

        assert "fen" in data_df.columns
        assert "moves" in data_df.columns
        assert "evaluation" in data_df.columns
        assert len(data_df.columns) == 3

        samples = [i for i in data_df.sort_index(axis=1).values.tolist()]

        self.samples = samples

    def __getitem__(self, index):
        evaluation, position, best_moves = self.samples[index]

        if evaluation is None:
            normalized_evaluation = torch.tensor([0.0], dtype=torch.float32)
        elif isinstance(evaluation, str):
            if evaluation[0] == "M" and evaluation[1] == "-":
                evaluation = -1000
            elif evaluation[0] == "M":
                evaluation = 1000

            normalized_evaluation = torch.tanh(
                torch.tensor([int(evaluation)], dtype=torch.float32) / 600
            )
        else:
            normalized_evaluation = torch.tensor(
                [float(evaluation)], dtype=torch.float32
            )

        decoded_position = tools.encode_fen_pos(position)

        if best_moves is None:
            encoded_best_moves = torch.zeros(20480, dtype=torch.float32)
        else:
            best_move_indices = [
                tools.encode_UCI_to_int(m.strip()) for m in best_moves.split(",")
            ]
            board = chess.Board(position)
            legal_move_indices = [
                tools.encode_UCI_to_int(m.uci()) for m in board.legal_moves
            ]
            encoded_best_moves = utils.create_policy_distribution_with_smoothing(
                best_move_indices, legal_move_indices
            )

        return decoded_position, encoded_best_moves, normalized_evaluation

    def __len__(self):
        return len(self.samples)

    # def __getitem__(self, index):
    #     position, best_moves, evaluation = self.samples[index]
    #     print(position, best_moves, evaluation)

    #     if evaluation is None:
    #         pass
    #     elif evaluation[0] == "M" and evaluation[1] == "-":
    #         evaluation = -1000
    #     elif evaluation[0] == "M":
    #         evaluation = 1000

    #     decoded_position = tools.encode_fen_pos(position)
    #     encoded_best_move = F.one_hot(
    #         torch.tensor(tools.encode_UCI_to_int(best_move), dtype=torch.int64),
    #         num_classes=20480,
    #     )
    #     normalized_evaluation = (
    #         torch.tensor([int(evaluation)], dtype=torch.float32) / 1000
    #     )  # Normalize to [-1, 1]

    #     return decoded_position, encoded_best_move, normalized_evaluation

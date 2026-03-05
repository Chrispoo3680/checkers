import chess
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..common import tools


class ChessDataset(Dataset):
    def __init__(self, data_df: pd.DataFrame):

        assert "fen" in data_df.columns
        assert "move" in data_df.columns
        assert "evaluation" in data_df.columns
        assert len(data_df.columns) == 3

        samples = [i for i in data_df.values.tolist()]

        self.samples = samples

    def __getitem__(self, index):
        position, best_move, evaluation = self.samples[index]

        if evaluation is None:
            normalized_evaluation = torch.empty(0)
        else:
            if evaluation[0] == "M" and evaluation[1] == "-":
                evaluation = -1000
            elif evaluation[0] == "M":
                evaluation = 1000

            normalized_evaluation = torch.tanh(
                torch.tensor([int(evaluation)], dtype=torch.float32) / 400
            )

        decoded_position = tools.encode_fen_pos(position)

        if best_move is None:
            encoded_best_move = torch.empty(0)
        else:
            epsilon = 0.1
            one_hot = F.one_hot(
                torch.tensor(tools.encode_UCI_to_int(best_move), dtype=torch.int64),
                num_classes=20480,
            ).float()

            board = chess.Board(position)
            legal_move_indices = [
                tools.encode_UCI_to_int(m.uci()) for m in board.legal_moves
            ]
            num_legal_moves = len(legal_move_indices)
            legal_moves_mask = torch.zeros(20480, dtype=torch.float32)
            legal_moves_mask[legal_move_indices] = 1.0

            encoded_best_move = (
                1 - epsilon
            ) * one_hot + epsilon * legal_moves_mask / num_legal_moves

        return decoded_position, encoded_best_move, normalized_evaluation

    def __len__(self):
        return len(self.samples)

    # def __getitem__(self, index):
    #     position, best_move, evaluation = self.samples[index]
    #     print(position, best_move, evaluation)

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

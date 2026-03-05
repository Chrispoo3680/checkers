import pandas as pd
import torch
from torch.utils.data import Dataset

from ..common import tools


class ChessDataset(Dataset):
    def __init__(self, data_path):

        df = pd.read_csv(data_path)

        samples = [i[1:] for i in df.values.tolist()]

        self.samples = samples

        self.data_path = data_path

    def __getitem__(self, index):
        position, best_move, evaluation = self.samples[index]

        if evaluation[0] == "M" and evaluation[1] == "-":
            evaluation = -1000
        elif evaluation[0] == "M":
            evaluation = 1000

        decoded_position = tools.decode_fen_pos(position)
        best_move = torch.tensor(tools.encode_USI_to_int(best_move), dtype=torch.int64)
        normalized_evaluation = (
            torch.tensor(int(evaluation), dtype=torch.float32) / 1000
        )  # Normalize to [-1, 1]

        return decoded_position, best_move, normalized_evaluation

    def __len__(self):
        return len(self.samples)

import pandas as pd
from torch import tanh
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

        decoded_position = tools.decode_fen_pos(position)
        normalized_evaluation = tanh(evaluation)  # Normalize to [-1, 1]

        return decoded_position, best_move, normalized_evaluation

    def __len__(self):
        return len(self.samples)

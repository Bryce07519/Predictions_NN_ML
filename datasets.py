import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import List 

class CryptoDataset(Dataset):
    """滑动窗口构造 (X, y)
    X: [lookback, num_features]
    y: 下一根 bar 收益率 (scalar)
    """

    def __init__(self, data: pd.DataFrame, lookback: int, feature_cols: list, target_col: str):
        self.X = data[feature_cols].values.astype("float32")
        self.y = data[target_col].shift(-1).values.astype("float32")
        # 去掉因 shift 产生的最后一个 NaN
        self.X, self.y = self.X[:-1], self.y[:-1]
        self.lookback = lookback

    def __len__(self):
        return len(self.X) - self.lookback + 1

    def __getitem__(self, idx):
        x = self.X[idx : idx + self.lookback]
        y = self.y[idx + self.lookback - 1]
        return torch.tensor(x), torch.tensor(y)
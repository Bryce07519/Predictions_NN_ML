import torch
import torch.nn as nn


class TimeSeriesTransformer(nn.Module):
    """简化版 Transformer Encoder 用于时间序列。"""

    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.input_linear = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_linear(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        return self.fc(x).squeeze(-1)

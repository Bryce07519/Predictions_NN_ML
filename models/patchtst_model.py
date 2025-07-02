try:
    from timesnet.models import PatchTST
except ImportError:
    PatchTST = None

import torch.nn as nn


class PatchTSTModel(nn.Module):
    """封装开源 PatchTST。"""

    def __init__(self, input_size: int, patch_len: int = 16, d_model: int = 128):
        super().__init__()
        if PatchTST is None:
            raise ImportError("Please install PatchTST implementation, e.g., `pip install timesnet-pytorch`.")
        self.model = PatchTST(
            c_in=input_size,
            patch_len=patch_len,
            stride=patch_len // 2,
            d_model=d_model,
            n_heads=4,
            dropout=0.1,
            n_layers=3,
            seq_len=Config.LOOKBACK,
            pred_len=1,
        )

    def forward(self, x):
        # PatchTST expects shape [B, C, T]
        x = x.transpose(1, 2)
        out = self.model(x)
        return out.squeeze(-1)
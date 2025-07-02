import numpy as np
import torch
from tqdm import tqdm

from utils.metrics import summarize_returns, directional_accuracy
from utils.plotting import plot_equity_curve


def backtest(model: torch.nn.Module, test_loader, device="cpu"):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Backtesting"):
            x = x.to(device)
            preds.append(model(x).cpu().numpy())
            targets.append(y.numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # 策略：按预测方向做多/做空 1 单位
    strategy_returns = np.sign(preds) * targets
    metrics = summarize_returns(strategy_returns)

    cum_ret = np.cumprod(1 + strategy_returns)
    plot_equity_curve(cum_ret, title="Backtest Equity Curve (Out of Sample)")

    return metrics
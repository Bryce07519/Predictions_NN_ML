import numpy as np


def directional_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    return np.mean((pred > 0) == (target > 0))


def max_drawdown(cumulative_returns: np.ndarray) -> float:
    roll_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - roll_max) / roll_max
    return drawdown.min()


def summarize_returns(returns: np.ndarray):
    cum_ret = np.cumprod(1 + returns)
    return {
        "Total Return": cum_ret[-1] - 1,
        "Max Drawdown": max_drawdown(cum_ret),
        "Win Rate": directional_accuracy(returns, np.zeros_like(returns)),
        "Mean Return": returns.mean(),
        "Std Return": returns.std(),
    }
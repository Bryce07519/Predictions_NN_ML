import numpy as np
import pandas as pd


def add_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    """示例量价特征：收益、振幅、VWAP、成交量变化率等。"""
    df = df.copy()
    df["log_return"] = np.log(df["close"].pct_change() + 1)
    df["high_low_range"] = (df["high"] - df["low"]) / df["low"]
    df["close_open_change"] = (df["close"] - df["open"]) / df["open"]
    df["vwap"] = (df["close"] * df["volume"]).rolling(window=20).sum() / df["volume"].rolling(window=20).sum()
    df["vol_change"] = df["volume"].pct_change()
    df.fillna(0, inplace=True)
    return df
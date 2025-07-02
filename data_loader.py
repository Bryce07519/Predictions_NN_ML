import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_raw_data(file_path: str) -> pd.DataFrame:
    """读取 Binance CSV，并将 time 列转换为 datetime 索引。"""
    df = pd.read_csv(file_path)
    if "timestamp" in df.columns:
        # df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
    return df


def preprocess(df: pd.DataFrame, feature_cols: list, scaler: StandardScaler = None):
    """基础预处理：填充缺失、标准化。只用训练集拟合 scaler，避免未来泄漏。"""
    df = df.copy()
    df.dropna(inplace=True)
    if scaler is not None:
        df[feature_cols] = scaler.transform(df[feature_cols])
    return df

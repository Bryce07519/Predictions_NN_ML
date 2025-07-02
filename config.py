class Config:
    DATA_PATH = "/Users/bryce/Documents/Code/Bryce/Data/Binance/futures/BTC/USDT_1m_all.csv"  # 本地数据路径
    LOOKBACK = 120  # 过去 N 个 bar 作为输入
    FEATURES = [
        "open", "high", "low", "close", "volume",
        # feature_engineering.py 会附加的衍生特征
    ]
    TARGET_SHIFT = 1  # 预测下一个 bar 收益

    # 训练参数
    BATCH_SIZE = 512
    EPOCHS = 50
    LR = 1e-3
    VALID_RATIO = 0.1  # 训练集内部留作验证的比例

    # 模型保存路径
    CKPT_DIR = "./checkpoints"

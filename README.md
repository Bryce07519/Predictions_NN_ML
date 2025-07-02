###  **BTC-1m 量价预测** 代码框架


| 任务环节       | 关键模块/函数                                        | 说明                                                                                                                                                                                                                             |
| ---------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **特征工程**   | `feature_engineering.add_ta_features`          | 演示经典量价因子（对数收益、振幅、VWAP、成交量变化…），可自行追加任何 alpha                                                                                                                                                                                    |
| **数据集构造**  | `datasets.CryptoDataset`                       | 滚动窗口产生 lookback × features 输入，严格按时间序列顺序切割，避免未来信息泄漏                                                                                                                                                                             |
| **三类模型**   | `models.*`                                     | - `LSTMModel` 👉 最基础<br>- `TimeSeriesTransformer` 👉 标准 Transformer 编码器<br>- `PatchTSTModel` 👉 参考 PatchTST (ICLR 2023) ([github.com][1])；如需更前沿亦可接入 iTransformer (ICLR 2024) ([openreview.net][2]) 或 TimesNet ([arxiv.org][3]) |
| **训练流程**   | `train.prepare_datasets` & `train.train_model` | ① 按 90 : 10 划分样本内/外，训练集再划 10 % 做验证 ② 早停+最优权重保存 ③ 训练记录保存在 `history`                                                                                                                                                             |
| **一步预测**   | `Config.TARGET_SHIFT = 1`                      | 目标定义为 *下一个* bar 的对数收益率                                                                                                                                                                                                         |
| **回测与可视化** | `backtest.backtest`                            | 输出胜率、累计收益、最大回撤等，并绘制 Equity Curve                                                                                                                                                                                               |

#### 如何使用

```bash
# 安装依赖
pip install pandas numpy scikit-learn torch matplotlib tqdm timesnet-pytorch  # PatchTST 依赖
# 运行
python main.py --model lstm          # 或 transformer / patchtst
```

* **数据格式** 请将 Binance 1-minute 蜡烛图 (OHLCV) CSV 文件路径写到 `config.py::DATA_PATH`；若时间戳字段名不同（示例用 `timestamp`），在 `data_loader.load_raw_data` 中调整即可。
* **扩展模型** 要尝试 TimesNet、iTransformer 等，只需在 `models/` 新增子模块并在 `models.__init__` 注册，训练脚本无需改动。
* **指标&图表** `utils/metrics` 和 `utils/plotting` 集中处理，可轻松添加 Sharpe、Calmar 等更多统计量或自定义可视化。

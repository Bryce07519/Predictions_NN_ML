import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from config import Config
from data_loader import load_raw_data, preprocess
from feature_engineering import add_ta_features
from datasets import CryptoDataset
from utils.metrics import directional_accuracy


def prepare_datasets() -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    df = load_raw_data(Config.DATA_PATH)
    df = add_ta_features(df)

    feature_cols = [col for col in df.columns if col != "log_return"]  # log_return 作为 target
    scaler = StandardScaler()

    # 划分索引
    total_len = len(df)
    train_end = int(total_len * 0.9)
    val_end = int(train_end * (1 - Config.VALID_RATIO))

    df_train = df.iloc[:val_end]
    scaler.fit(df_train[feature_cols])

    # 预处理
    df_pre = preprocess(df, feature_cols, scaler)
    dataset = CryptoDataset(df_pre, Config.LOOKBACK, feature_cols, target_col="log_return")

    train_len = val_end - Config.LOOKBACK + 1
    val_len = train_end - val_end
    test_len = len(dataset) - train_len - val_len

    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=Config.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, scaler


def train_model(model: torch.nn.Module, train_loader, val_loader, optimizer, criterion, device="cpu"):
    best_val_loss = np.inf
    history = []
    for epoch in range(Config.EPOCHS):
        model.train()
        train_losses = []
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses, val_acc = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_losses.append(loss.item())
                val_acc.append(directional_accuracy(pred.cpu().numpy(), y.cpu().numpy()))
        mean_val_loss = np.mean(val_losses)
        history.append({"epoch": epoch, "train_loss": np.mean(train_losses), "val_loss": mean_val_loss, "val_dir_acc": np.mean(val_acc)})

        # 早停
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            Path(Config.CKPT_DIR).mkdir(exist_ok=True, parents=True)
            torch.save(model.state_dict(), os.path.join(Config.CKPT_DIR, "best_model.pt"))

    return history


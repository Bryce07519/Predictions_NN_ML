import torch
from torch import optim, nn

from config import Config
from train import prepare_datasets, train_model
from backtest import backtest
from models import LSTMModel, TimeSeriesTransformer, PatchTSTModel


def run(model_name: str = "lstm"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, test_loader, _ = prepare_datasets()

    input_size = train_loader.dataset[0][0].shape[1]
    if model_name == "lstm":
        model = LSTMModel(input_size)
    elif model_name == "transformer":
        model = TimeSeriesTransformer(input_size)
    elif model_name == "patchtst":
        model = PatchTSTModel(input_size)
    else:
        raise ValueError("Unknown model name")

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    criterion = nn.MSELoss()

    history = train_model(model, train_loader, val_loader, optimizer, criterion, device)
    metrics = backtest(model, test_loader, device)

    print("Backtest metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return history, metrics
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm", "transformer", "patchtst"], default="lstm")
    args = parser.parse_args()
    run(args.model)

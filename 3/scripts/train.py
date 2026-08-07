import os

import torch
import torch.nn as nn

import logging

from utils.metrics import abs_error_sum


logger = logging.getLogger(__name__)


def eval_model(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    total_mae = 0
    n = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).view(-1, 1)

            out = model(x)
            loss = criterion(out, y)

            total_loss += loss.item() * x.size(0)
            total_mae += abs_error_sum(out, y)

            n += x.size(0)

    return total_loss / n, total_mae / n


def train_model(model, train_loader, val_loader, cfg, device):
    model = model.to(device)

    criterion = nn.MSELoss()

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(params, lr=cfg["lr"])

    best_mae = 100000

    save_path = cfg["save_path"]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    logger.info("training started")

    for epoch in range(cfg["epochs"]):
        model.train()

        train_loss = 0
        n = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).view(-1, 1)

            optimizer.zero_grad()

            out = model(x)
            loss = criterion(out, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            n += x.size(0)

        train_loss = train_loss / n

        val_loss, val_mae = eval_model(model, val_loader, criterion, device)

        logger.info("epoch %d: train_loss=%.4f | val_loss=%.4f | val_mae=%.2f",epoch + 1,train_loss,val_loss,val_mae)

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), save_path)

            logger.info("saved best model with val_mae=%.2f", best_mae)

    logger.info("training finished")

    return model
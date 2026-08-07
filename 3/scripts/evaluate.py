import torch

import logging

from utils.metrics import abs_error_sum


logger = logging.getLogger(__name__)


def evaluate(model, loader, device):
    model.eval()

    total_mae = 0
    n = 0

    logger.info("evaluation started")

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).view(-1, 1)

            out = model(x)

            total_mae += abs_error_sum(out, y)
            n += x.size(0)

    mae = total_mae / n

    logger.info("final MAE: %.2f", mae)

    return mae
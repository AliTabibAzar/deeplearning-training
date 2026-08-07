import yaml
import torch
import logging
import logging.config

from data.data_loader import get_data
from models.model import make_model
from scripts.train import train_model
from scripts.evaluate import evaluate


def setup_logging():
    with open("config/logging.yaml", "r", encoding="utf-8") as f:
        log_cfg = yaml.safe_load(f)

    logging.config.dictConfig(log_cfg)


def main():
    setup_logging()

    logger = logging.getLogger(__name__)

    with open("config/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: %s", device)

    train_loader, val_loader = get_data(cfg["data"])
    logger.info("dataloaders are ready")

    model = make_model(cfg["model"]["name"])
    logger.info("model created: %s", cfg["model"]["name"])

    model = train_model(model, train_loader, val_loader, cfg["train"], device)

    evaluate(model, val_loader, device)

    logger.info("finished")


if __name__ == "__main__":
    main()
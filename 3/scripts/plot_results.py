import sys
import logging
import logging.config
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from utils.visualization import plot_from_log


def setup_logging():
    with open(PROJECT_ROOT / "config" / "logging.yaml", "r", encoding="utf-8") as f:
        log_cfg = yaml.safe_load(f)
    logging.config.dictConfig(log_cfg)


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    log_path = PROJECT_ROOT / "training.log"
    save_path = PROJECT_ROOT / "training_curves.png"
    
    logger.info("Generating plots from training log...")
    
    success = plot_from_log(str(log_path), str(save_path))
    
    if success:
        logger.info("Done! Check: %s", save_path)
    else:
        logger.error("Failed to generate plots")


if __name__ == "__main__":
    main()
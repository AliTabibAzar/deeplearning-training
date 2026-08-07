import torch
import yaml
import logging
import logging.config
import os
from data.data_loader import get_dataloader
from models.model import Generator, Discriminator, weights_init
from scripts.train import train
from utils.visualization import create_gif

# setup logger
with open("config/logging.yaml", "r") as f:
    log_cfg = yaml.safe_load(f)
logging.config.dictConfig(log_cfg)
logger = logging.getLogger(__name__)

def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() and config['device'] == "cuda" else "cpu")
    logger.info(f"Using device: {device}")

    dataloader = get_dataloader(
        config['dataset_path'], 
        config['image_size'], 
        config['batch_size'],
        config.get('max_images', None)
    )
    logger.info(f"Data loaded. Batches per epoch: {len(dataloader)}")
    
    netG = Generator(config['nz'], config['ngf'], 3).to(device)
    netG.apply(weights_init)
    
    netD = Discriminator(config['ndf'], 3).to(device)
    netD.apply(weights_init)

    train(netG, netD, dataloader, config, device)
    
    logger.info("Creating GIF...")
    os.makedirs("outputs", exist_ok=True)
    create_gif("outputs/gif_frames", "outputs/training_progress.gif")
    logger.info("Done!")

if __name__ == "__main__":
    main()
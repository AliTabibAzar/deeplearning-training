import torch
import yaml
import os
from models.model import Generator
from utils.visualization import save_image_grid

def evaluate():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    netG = Generator(config['nz'], config['ngf'], 3).to(device)
    netG.load_state_dict(torch.load("models/saved_models/netG_final.pth", map_location=device))
    netG.eval()
    
    with torch.no_grad():
        noise = torch.randn(16, config['nz'], 1, 1, device=device)
        fake = netG(noise).detach().cpu()
        
    os.makedirs("outputs", exist_ok=True)
    save_image_grid(fake, "outputs/eval_result.png", nrow=4)
    print("Saved to outputs/eval_result.png")

if __name__ == "__main__":
    evaluate()
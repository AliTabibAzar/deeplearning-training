import torch
import torch.nn as nn
import torch.optim as optim
import os
import logging
from utils.visualization import save_image_grid

logger = logging.getLogger(__name__)

def train(netG, netD, dataloader, config, device):
    criterion = nn.BCELoss()
    
    optimizerG = optim.Adam(netG.parameters(), lr=config['lr'], betas=(config['beta1'], 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=config['lr'], betas=(config['beta1'], 0.999))

    fixed_noise = torch.randn(64, config['nz'], 1, 1, device=device)
    real_label = 1.0
    fake_label = 0.0
    
    gif_folder = "outputs/gif_frames"
    os.makedirs(gif_folder, exist_ok=True)

    for epoch in range(config['epochs']):
        for i, data in enumerate(dataloader):
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            
            noise = torch.randn(b_size, config['nz'], 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            errD = errD_real + errD_fake
            optimizerD.step()

            netG.zero_grad()
            label.fill_(real_label) # fake labels are real for G
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()

            if i % 100 == 0:
                logger.info(f"[{epoch}/{config['epochs']}][{i}/{len(dataloader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}")
        
        with torch.no_grad():
            fake_fixed = netG(fixed_noise).detach().cpu()
        save_image_grid(fake_fixed, os.path.join(gif_folder, f"epoch_{epoch:02d}.png"))
        
        os.makedirs("models/saved_models", exist_ok=True)
        torch.save(netG.state_dict(), "models/saved_models/netG_final.pth")
        torch.save(netD.state_dict(), "models/saved_models/netD_final.pth")
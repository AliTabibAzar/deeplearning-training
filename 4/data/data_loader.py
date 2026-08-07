import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_dataloader(data_dir, img_size=64, batch_size=128, max_images=None):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    # limit dataset size to speed up training
    if max_images and max_images < len(dataset):
        dataset = Subset(dataset, range(max_images))
        
    # drop_last=True to avoid batch size 1 issues with BatchNorm
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    return dataloader
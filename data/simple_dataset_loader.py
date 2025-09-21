
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from typing import Optional

def load_cifar_dataset(cfg=None, img_size=8, batch_size=2, subset_size=None, only_return_dataset=False):  # False, True, "both", 
    """
    Simple cifar10 loader.
    
        Returns: 
            dataloader
            dataset: if return_dataset = True
    """
    if cfg is not None:
        img_size=cfg.model.image_size
        batch_size=cfg.training.batch_size

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5, 0.5, 0.5))
    ])

    dataset = torchvision.datasets.CIFAR10(root=".", train=True, download=True, transform=transform)
    if subset_size is not None and subset_size < len(dataset):
        dataset = Subset(dataset, list(range(subset_size)))
    if only_return_dataset == True:
        return dataset

    dataloader = DataLoader(dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,              # good default for tests / Windows
        persistent_workers=False,   # avoids the "needs num_workers > 0" warning
        pin_memory=False,   
    )

    if only_return_dataset == "both":
        return dataloader, dataset
    
    return dataloader



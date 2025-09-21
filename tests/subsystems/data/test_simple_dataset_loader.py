
import pytest
from data.simple_dataset_loader import load_cifar_dataset
from omegaconf import OmegaConf

def test_simple_dataset_loads():

    cfg = OmegaConf.create({
        "training":
            {
                "batch_size": 2,
                },
        "model":
            {
                "image_size": 16,
        },
    })
    dataloader, dataset = load_cifar_dataset(cfg, only_return_dataset="both")

    #print(dataloader)
    print(dataset)

    assert dataset, dataloader is not None
    

    # quick sanity load a few

    for batch in dataloader:
        assert batch is not None
        



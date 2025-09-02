
import torchvision.transforms as T


def build_transforms(cfg, mode="train"):
    # Build image transforms based on config. 

    if mode == "train":
        augmentations = [
            T.RandomResizedCrop(cfg.data.image_size, scale=(0.8, 1.0)),  # not in data cfg!   # a random resized crop?
            T.RandomHorizontalFlip(),
        ]
    else:   # validation / test transforms
        augmentations = [
            T.Resize(cfg.data.image_size + 32),
            T.CenterCrop(cfg.data.image_size),
        ] 

    augmentations.append(T.ToTensor())
    augmentations.append(T.Normalize(mean=[0.5], std=[0.5]))

    transform = T.Compose(augmentations)
    return transform

    # Very simple, can be extended
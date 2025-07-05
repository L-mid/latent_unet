
import torch
from model.config import load_config
from model.build_unet import build_unet_from_config

if __name__ == "__main__":
    # Load the config!
    cfg = load_config("configs/unet_config.yaml")

    # Build the model
    model = build_unet_from_config(cfg)

    # Print summary
    print("\nU-Net Model Built Sucessfully!")
    print(model, "Is the model.")

    # Optionally: run a dummy forward pass
    x = torch.randn(1, cfg.model.in_channels, cfg.model.image_size, cfg.model.image_size)
    t = torch.randint(0, cfg.schedule.timesteps, (1,))

    with torch.no_grad():
        y = model(x, t)
    print(f"Forward pass sucess! Output shape: {y.shape}.")

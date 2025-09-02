
# Latent U-Net Diffusion

A modular, scalable, aggressively optimized U-Net framework for training denoising diffusion models (DDPM, DDIM, etc.) on 256x256+, resolution images, even on limited compute.

Supports plug-and-play experimentation with residual blocks, time embeddings, attention mechanisims, (flash, windowed), midblocks, schedulers, samplers, +.

--- Features

- config driven with 'OmegaConf'.
- Gradient checkpointing, AMP, EMA
- Modular U-Net blocks (residuals, attention, etc.)
- Flash attention / Swin-style window attention support.
- Supports DDPM and DDIM samplers.
- Scalable to latent diffusion or larger image sizes.
- Plug-in loss functions (p2, MSE, etc).
- Sample visualizations and training logging built in.

---

## Installation

```bash
git clone https://github.com/L-Mid/latent_unet.git
cd latent-unet
pip install -e .


## Configuration

Edit configs/unet_config.yaml to control:
- Model structure (ch_mults, attention kind, block types)
- Training hyperparameters (batch_size, EMA, checkpointing)
- Sampler type (ddpm, ddim)
- Optimization details (lr, amp, grad_clip)


## Training!

Training the model (from scratch):
python run.py --config configs/unet_config.yaml     # what is this command structure used?

Training the model (resuming training from checkpoint):
python run.py --config configs/unet_config.yaml --resume


## Sampling

Generated images will appear in outputs/ :
python run.py --config configs/unet_config.yaml --sample_only


## Logging
- TensorBoard logging to ./logs/
- Add your wandb API key and enable via logger.py if needed.


## TODO
- Add LoRA for lightweight fine-tuning
- Integrate VAE encoder/decoder for latent diffusion
- Add DDP support for multi-GPU scaling
- Add more attention types (linear, Performer, etc.)


## License
MIT License


## Citations
This repo draws inspiration from:
- [Ho et al., 2020] Denoising Diffusion Probabilistic Models (DDPM)
- [Song et al., 2021] DDIM
- [Rombach et al., 2022] Latent Diffusion Models


---------------
# Project structure:

latent_unet/
|
├── tbd


## For mapping tree keys:

indentations, 
or use (copy paste):

├──

└──

│



# Happy training!




























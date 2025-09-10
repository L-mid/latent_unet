
# Setup:

"""
Dataset: CIFAR-10 (32x32, 50k images, easy to download)
Resolution: 32x32.
Batch size: 64 (or as fits your GPU).
Model: A tiny U-Net.
    Channels: 64 base, 128, 256.
    1-2 ResBlocks per stage.
    Simple sinusoidal timestep embedding.
    Attention only at 16x16 resolution.
Timesteps: 1000 betas, cosine or linear schedule.
Optimizer: AdamW, lr=2e-4, weight decay=0.
EMA: decay=0.999.
Training length: 100k-200k steps (enough to see shapes form)

"""










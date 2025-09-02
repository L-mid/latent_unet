
import os
import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from einops import rearrange
import numpy as np

# === NOTES:
"""
Extremely messy, different output dirs everywhere, not tested if all can be visualized together either.
"""


VISUALIZER_REGISTRY = {}

def register_visualizer(name):
    def decorator(func):
        VISUALIZER_REGISTRY[name] = func
        return func     # helpful so you can still call it directly if you want
    return decorator



def _save_image_grid(tensors, save_dir, filename, nrow=4, normalize=True):
    os.makedirs(save_dir, exist_ok=True)
    grid = vutils.make_grid(tensors, nrow=nrow, normalize=normalize, scale_each=True)
    vutils.save_image(grid, os.path.join(save_dir, filename))

@register_visualizer("noising")
def compare_noising(*, x0, xt, xhat, step, save_path, max_images=None, **_):
    m = max_images or min(x0.size(0), xt.size(0), xhat.size(0))
    imgs = torch.cat([x0[:m], xt[:m], xhat[:m]], dim=0)
    _save_image_grid(imgs, save_dir=save_path, filename=f"denoising_step{step:06}.png")

@register_visualizer("attention")
def visualize_attention(*, attn_maps, step, save_path, **_):
    # visualize attention maps per head
    for i, attn in enumerate(attn_maps): # [B, Heads, H, W]
        B, H, W = attn.shape[0], attn.shape[-2], attn.shape[-1]
        fig, axs = plt.subplots(1, B, figsize=(B * 2, 2))
        for j in range(B):
            axs[j].imshow(attn[j].mean(0).cpu().detach().numpy(), cmap="viridis")
            axs[j].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"attn_layer{i}_step{step:06}.png"))
        plt.close()

@register_visualizer("time_embeddings")
def plot_timestep_embedding(*, t_embeds, step, save_path, **_):
    # PCA projection of time embeddings.
    from sklearn.decomposition import PCA
    t_embeds = t_embeds.cpu().detach().numpy()
    pca = PCA(n_components=2)
    proj = pca.fit_transform(t_embeds)
    plt.figure(figsize=(6, 6))
    plt.scatter(proj[:, 0], proj[:, 1], c=np.arange(len(proj)), cmap="viridis", s=10)
    plt.colorbar(label="Timestep")
    plt.title("Time Embedding PCA")
    plt.savefig(os.path.join(save_path, f"time_embedding_pca_{step:06}.png"))
    plt.close()

@register_visualizer("schedule") 
def plot_beta_schedule(*, betas, name="schedule", save_path, **_):
    # Plot beta (or alpha) schedules over time.
    plt.figure()
    plt.plot(betas.cpu().numpy(), label="Beta")
    plt.title("Beta Schedule"); plt.xlabel("Timestep"); plt.ylabel("Beta"); plt.legend()
    plt.savefig(os.path.join(save_path, f"{name}.png"))
    plt.close()

@register_visualizer("guidance")
def plot_guidance_effects(*, samples, guidance_scales, step, save_path, **_):
    # Visualize samples under different guidance strengths.
    imgs = rearrange(samples, 'g b c h w -> (g b) c h w')
    _save_image_grid(imgs, save_dir=save_path, filename=f"guidance_grid_{step:06}.png", nrow=len(guidance_scales))


@register_visualizer("grad_flow")
def plot_grad_flow(named_parameters, save_path="grid_flow.png", **_):
    # Visualize gradient magnitudes across the model layers.
    # Great for checking vanishing or exploding gradients.
    ave_grads = []
    max_grads = []
    layers = []

    for n, p in named_parameters:
        if p.requires_grad and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
            max_grads.append(p.grad.abs().max().item())

    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, color="c", label='Max Grad')
    plt.bar(np.arange(len(ave_grads)), ave_grads, alpha=0.5, color="b", label='Mean Grad')
    plt.hlines(0, 0, len(ave_grads), linewidth=1.5, color="k")
    plt.xticks(range(len(ave_grads)), layers, rotation="vertical", fontsize=8)
    plt.title("Gradient Flow per Layer")
    plt.xlabel("Layers")
    plt.ylabel("Gradient Mangitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


@register_visualizer("param_norms")
def plot_param_norms(model, save_path="param_norms.png"):
    # Visualize L2 norms of model parameters per layer
    # Useful for debugging abnormal parameter magnitudes.

    norms = []
    names = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            norms.append(param.data.norm().item())
            names.append(name)


    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(norms)), norms, color='orange')
    plt.xticks(range(len(norms)), rotation='vertical', fontsize=8)
    plt.title("Weight Norms Per Layer")
    plt.xlabel("Layer")
    plt.ylabel("L2 Norm")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


@register_visualizer("latents")
def visualize_latents(latents, step=None, save_path="latents", **_):
    # Visualizes a grid of latent features (e.g., from encoder bottleneck).
    # Expects latents of shape [B, C, H, W] and saves first few channels.
    out = os.path.join(save_path, "latents") # put subdir under tmpdir

    os.makedirs(out, exist_ok=True)
    b, c, h, w = latents.shape
    n = min(b, c)

    for i in range(n):
        img = latents[0, i].detach().cpu().numpy()
        plt.imshow(img, cmap='viridis')
        plt.axis("off")
        plt.title(f"Latent ch {i}")
        plt.savefig(os.path.join(save_path, f"latent_ch_{i}_step{step}.png"))
        plt.close()

def visualize_everything(
    model, 
    latents=None,
    step=None,
    named_parameters=None,
    cfg=None,
    betas=None,
):
    if not cfg or not cfg.visualization.enabled:
        return

    save_path = cfg.visualization.output_dir
    os.makedirs(save_path, exist_ok=True)

    base_kwargs = {
        "model": model,
        "latents": latents,
        "params": named_parameters,
        "step": step,
        "betas": betas,
        "save_path": cfg.visualization.output_dir,
        "n_channels": 4                             # was cfg.latent_channels before? 
    }

    for name in cfg.visualization.use:
        fn = VISUALIZER_REGISTRY.get(name)
        if not fn:
            continue
        try:
            fn(**base_kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to run visualizer '{name}': {e}")
















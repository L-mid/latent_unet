
import torch
import torch.nn.functional as F
from tqdm import tqdm
from diffusion.sampler_registry import register_sampler


# Flexible schedule based on Karras et al. (rho-schedule)
def get_edm_schedule(num_steps, sigma_min, sigma_max, rho, device):
    ramp = torch.linspace(0, 1, num_steps, device=device)
    sigmas = (sigma_max**(1/rho) + ramp * (sigma_min**(1/rho) - sigma_max**(1/rho)))**rho
    return sigmas

# Input normalization constants
def edm_preprocess(x, sigma):
    c_skip = 1 / torch.sqrt(1 + sigma ** 2)
    c_out = sigma / torch.sqrt(1 + sigma ** 2)
    c_in = 1 / torch.sqrt(sigma ** 2 + 1)
    return c_skip, c_out, c_in

# Classifier-free guidance application
def apply_cfg(x, pred, guidance_scale, cond_pred=None):
    return pred if cond_pred is None else pred + guidance_scale * (cond_pred - pred)


@register_sampler("edm")
# Full EDM sampling procedure
@torch.no_grad()
def edm_sample(
    model, 
    shape,
    cfg, 
    condition_fn=None,
    device="cuda"
):
    

    # Load from YAML config
    num_steps   = cfg.sampler.num_steps
    sigma_min   = cfg.sampler.sigma_min
    sigma_max   = cfg.sampler.sigma_max
    rho         = cfg.sampler.rho
    solver      = cfg.sampler.solver
    guidance_scale = getattr(cfg.sampler, "guidance_scale", 0.0)
    classifier_free = cfg.sampler.get("classifier_free", False)

    sigmas = get_edm_schedule(num_steps=num_steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho, device=device)
    x = torch.randn(shape, device=device) * sigmas[0] # Sample from largest noise

    for i in tqdm(range(len(sigmas) - 1)):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        t = sigma.expand(shape[0])

        if classifier_free:
            x_in = torch.cat([x] * 2, dim=0)
            t_in = torch.cat([t] * 2, dim=0)
            out = model(x_in, t_in)
            pred, cond_pred = out.chunk(2, dim=0)
            out = apply_cfg(x, pred, guidance_scale, cond_pred)
        else:
            out = model(x, t)

        d = (x - out) / sigma
        dt = sigma_next - sigma

        if solver == "euler":
            x = x + d * dt
        elif solver == "heun":
            x_pred = x + d * dt
            t_next = sigma_next.expand(shape[0])
            if classifier_free:
                x_in_next = torch.cat([x_pred] * 2, dim=0)
                t_in_next = torch.cat([t_next] * 2, dim=0)
                out_next = apply_cfg(x_pred, pred, guidance_scale, pred)
            else: 
                out_next = model(x_pred, t_next)
                d_next = (x_pred - out_next) / sigma_next
                x = x + 0.5 * (d + d_next) * dt  
        else: raise ValueError(f"Unknown solver type: {solver}")

        return x









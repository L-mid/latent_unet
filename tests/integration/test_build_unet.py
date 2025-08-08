
import torch
import pytest
from model.build_unet import build_unet_from_config

from modules.down_block import DownBlock
from modules.up_block import UpBlock
from modules.mid_block import MidBlock
from modules.final_head import FinalHead
from model.unet import UNet

# ---------------------------------------------------------
# Shared Fixture: Real UNet Model from Config
# ---------------------------------------------------------
@pytest.fixture(scope="module")
def model_and_config(unet_config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_unet_from_config(unet_config).to(device) #
    model.eval()
    return model, unet_config # cuda issue with devices in resblock and here


# ---------------------------------------------------------
# Basic Model-Level Integration
# ---------------------------------------------------------
class TestUNetIntegration:
    def test_forward_pass_runs(self, model_and_config):
        model, cfg = model_and_config
        x = torch.randn(2, cfg.model.in_channels, 32, 32)
        t = torch.randint(0, 1000, (2,))
        out = model(x, t)

        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    
    def test_backward_pass_runs(self, model_and_config):
        model, cfg = model_and_config
        x = torch.randn(2, cfg.model.in_channels, 32, 32, requires_grad=True)
        t = torch.randint(0, 1000, (2,))
        out = model(x, t)
        loss = out.mean()
        loss.backward()

        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert any(g is not None and g.abs().sum() > 0 for g in grads), "No gradients flowed"


    def test_shape_trace(self, model_and_config):
        model, cfg = model_and_config
        x = torch.randn(1, cfg.model.in_channels, 32, 32)
        t = torch.randint(0, 1000, (1,))

        print("\n--- Unet Shape Trace ---")

        def hook_fn(mod, inp, out):
            name = mod.__class__.__name__
            shape = out.shape if isinstance(out, torch.Tensor) else tuple(o.shape for o in out)
            print(f"[TRACE] {name}: {shape}")

        # Register trace for top-level submodules
        model.mid.register_forward_hook(hook_fn)
        model.final_head.register_forward_hook(hook_fn)

        for i, block in enumerate(model.downs):
            block.register_forward_hook(lambda m, i, o: print(f"[Down {i}] {m.__class__.__name__}: {o.shape if not isinstance(o, tuple) else tuple(x.shape for x in o)}"))

        for i, block in enumerate(model.ups):
            block.register_forward_hook(lambda m, i, o: print(f"[Up {i}] {m.__class__.__name__}: {o.shape if not isinstance(o, tuple) else tuple(x.shape for x in o)}"))

        _ = model(x, t)

# ----------------------------------------------------------------------
# Skip Connection Logic 
# ----------------------------------------------------------------------
class TestSkipShapeFlow:
    def test_skip_shapes_match(self, model_and_config):
        model, cfg = model_and_config
        x = torch.randn(1, cfg.model.in_channels, 64, 64)
        t = torch.randint(0, 1000, (1,))

        skip_shapes = []

        def record_skip_shapes(mod, inp, out):
            if isinstance(out, tuple) and len(out) > 1:
                skip_shapes.append(out[1].shape)
            else:
                skip_shapes.append(out.shape)

        for mod in model.downs:
            mod.register_forward_hook(record_skip_shapes)

        model(x, t)

        print("\n--- Recorded Skip Shapes ---")
        for i, shape in enumerate(skip_shapes):
            print(f"DownBlock {i}: skip shape = {shape}")

        for i, (up, skip_shape) in enumerate(zip(model.ups, reversed(skip_shapes))):
            assert skip_shape is not None
            # Optional: check spatial size or channel count
            h, w = skip_shape[2:]
            assert h > 1 and w > 1, f"Invalid skip shape: {skip_shape} at UpBlock {i}"

            assert up.skip_channels == skip_shape[1], f"Expected skip channels {up.skip_channels}, got {skip_shape[1]}"

        
  

    def test_channel_config_summary(self, model_and_config):
        model, _ = model_and_config
        print("\n--- Channel Summary ---")
        for i, block in enumerate(model.downs):
            print(f"DownBlock {i}: in_ch={block.in_ch}, out_ch={block.out_ch}")
        for i, block in enumerate(model.ups):
            print(f"UpBlock {i}: in_ch={block.in_ch}, out_ch={block.out_ch}, skip_ch={block.skip_channels}")


# -----------------------------------------------------------------------------
# Time Embedding Check
# -----------------------------------------------------------------------------

def test_time_embedding_output_shape(model_and_config):
    model, cfg = model_and_config
    t = torch.randint(0, 1000, (4,))
    emb = model.time_embedding(t)

    expected_dim = cfg.time_embedding.params.dim
    assert emb.shape == (4, expected_dim), f"Expected (4, {expected_dim}), got {emb.shape}"
    assert torch.isfinite(emb).all(), "Time embedding contains non-finite values"


# -----------------------------------------------
# Attention/ResBlock Variant Coverage
# -----------------------------------------------
class TestAttentionVariantIntegration:
    @pytest.mark.parametrize("attn_type", ["none", "vanilla", "window", "flash"])
    def test_resblock_with_attention_variants_runs(self, attn_type, unet_config):
        cfg = unet_config.copy()
        cfg.attention.type = attn_type

        model = build_unet_from_config(cfg)
        model.eval()

        x = torch.randn(2, cfg.model.in_channels, 32, 32)
        t = torch.randint(0, 1000, (2,))
        out = model(x, t)

        assert out.shape == x.shape
        assert torch.isfinite(out).all()


    @pytest.mark.parametrize("attn_type", ["vanilla", "window", "flash"])
    def test_attention_changes_output(self, attn_type, unet_config):
        from copy import deepcopy
        # Deepcopy to isolate mutation of cfg
        cfg_attn = deepcopy(unet_config)
        cfg_none = deepcopy(unet_config)

        # Set attention type for each version
        cfg_attn.attention.kind = attn_type
        cfg_none.attention.kind = "none"     

        model_with_attn = build_unet_from_config(cfg_attn).eval()
        model_without_attn = build_unet_from_config(cfg_none).eval()

        x = torch.randn(1, cfg_attn.model.in_channels, 32, 32)
        t = torch.randint(0, 1000, (1,))

        out_with_attn = model_with_attn(x, t)
        out_no_attn = model_without_attn(x, t)

        assert not torch.allclose(out_with_attn, out_no_attn), f"{attn_type} attention had no effect on output!"


    def test_model_runs_on_cuda(self, model_and_config):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")


        model, cfg = model_and_config
        model = model.cuda()

        def _cpu_params(model):
            bad = []
            for name, module in model.named_modules():
                for pname, p in module.named_parameters(recurse=False):
                    if p.device.type != 'cuda':
                        bad.append(f"{name}.{pname}")
            return bad

        left = _cpu_params(model)
        print(f"Params left on CPU: {left}")

        x = torch.randn(1, cfg.model.in_channels, 32, 32).cuda()
        t = torch.randint(0, 1000, (1,)).cuda()

        out = model(x, t)
        assert out.is_cuda




"""
latent_unet_v1/
├── run.py 🔹
├── README.md 🔹
├── pyproject.toml 🔹
├── .gitignore 🔹
├── .github/workflows/ci.yaml 🔹

├── configs/
│   ├── unet_config.yaml 🔹
│   ├── test_config.yaml 🔹
│   └── mock_data_config.yaml 🔹

├── model/
│   ├── config.py 🔹
│   ├── build_unet.py 🔹
│   └── unet.py 🔹 

├── modules/
│   ├── time_embedding.py 🔹
│   ├── residual_block.py 🔹
│   ├── mid_block.py 🔹
│   ├── down_block.py 🔹
│   ├── up_block.py 🔹
│   ├── final_head.py 🔹
│   ├── norm_utils.py 🔹
│   └── attention/ 
│       ├── base_attention.py 🔹
│       ├── vanilla_attention.py 🔹
│       ├── window_attention.py 🔹
│       └── flash_attention.py 🔹

├── diffusion/ 🔹
│   ├── ddpm.py 🔹
│   ├── ddim.py 🔹
│   ├── edm.py 🔹
│   ├── sampler_registry.py 🔹
│   ├── sampler_utils.py 🔹
│   ├── forward_process.py 🔹
│   └── schedule.py 🔹

├── trainer/
│   ├── train_loop.py XXX (working on)
│   ├── cluster_utils.py 🔹
│   ├── logger.py 🔹
│   ├── optim_utils.py 🔹
│   ├── ema_utils.py 🔹
│   └── losses.py 🔹

├── data/
│   ├── loader.py 🔹
│   ├── mock_dataset.py 🔹
│   ├── (some others) 🔹
│   └── dataset_registry.py 🔹

├── utils/
│   ├── debug.py 🔹
│   ├── visualizer.py 🔹
│   ├── memory_tools.py
│   ├── failure_injection.py
│   ├── tensor_inspect.py
│   ├── vanilla_checkpointing.py 🔹
│   │
│   ├── zarr_checkpointing/ 🔹
│   │    ├── zarr_core.py 🔹
│   │    └── zarr_wrapper.py 🔹
│   │
│   └── tensorstore_checkpointing/
│       ├── __init__.py 🔹
│       ├── tensorstore_core.py 🔹
│       ├── tensorstore_wrapper.py 🔹
│       ├── schema_utils.py 🔹
│       ├── chunk_tuner.py 🔹
│       ├── registry.py 🔹
│       └── remote_utils.py 🔹

├── helpers/
│   ├── mock_configs.py
│   ├── fake_model.py
│   ├── fake_data.py
│   └── test_utils.py 🔹

├── tests/
│   ├──  conftest.py 🔹
│   │
│   ├── unit/
│   │   ├── test_time_embedding.py 🔹
│   │   ├── test_residual_block.py 🔹
│   │   ├── test_attention_block.py 🔹
│   │   ├── test_midblock.py 🔹
│   │   ├── test_down_block.py 🔹
│   │   ├── test_up_block.py 🔹
│   │   └── test_final_head.py 🔹
│   │
│   ├── integration/
│   │   ├── test_unet.py 🔹
│   │   ├── test_integrated_forward_process.py ?
│   │   ├── test_train_loop.py 🔹
│   │   └── test_ddpm_ddim_edm.py🔹
│   │
│   ├── end_to_end/
│   │   ├── test_cuda_health.py
│   │   ├── test_training_pipeline.py
│   │   └── test_sampling_loop.py
│   │
│   ├── subsystems/                
│   │   ├── checkpointing/
│   │   │   ├── test_tensorstore.py 🔹
│   │   │   ├── test_tensorstore_schema_utils.py 🔹
│   │   │   ├── test_zarr.py 🔹
│   │   │   └── test_vanilla_checkpointing.py 🔹
│   │   ├── data/
│   │   │   └── test_data_pipeline.py 🔹
│   │   ├── config/
|   |   |   ├── test_config_loading.py🔹
│   │   │   ├── test_config_integration.py
│   │   │   └── test_config_roundtrip.py🔹
│   │   │
│   │   ├── diffusion/
│   │   │   ├── test_schedule.py 🔹
│   │   │   ├── test_forward_process.py 🔹
│   │   │   ├── test_ddpm.py 🔹
│   │   │   ├── test_ddim.py 🔹
│   │   │   ├── test_edm.py 🔹
│   │   │   ├── test_losses.py 🔹
│   │   │   └── test_sampler_registry.py 🔹
│   │   │
│   │   ├── test_visualizer.py 🔹
│   │   ├── test_debug_hooks.py 
│   │   └── test_failure_injection.py
│   │
│   ├── regression/
│   │   └── test_sequential_nan_bug.py  # (placeholder for when)
│   │
│   └── __init__.py 





Legend:
🔹: Implimented once - (copied direct)                      - You typed it in, but it's "dead muscle memory".
🔺: Known            - (wrote with some copying)            - Can explain dataflow, run small tests, still peaks for APIs.
🔸: Level 1          - (can write without help)             - Can re-impliment with <3 peaks per function, tests pass.
🔸🔸: Level 2        - (can write and change it a bit)     - Can re-impliment model+train loop cold; contract tests all pass.       
🔸🔸🔸: Level 3     - (can write and refactor)             - Can inject bugs and your asserts/localization catch them quickly; can swap varients easily without breaking.
🌟: Mastered         - (automatic)                          - Can design new abstractions, refactor repo-scale systems, anticipate faliure modes easily.

~82 files (not updated)

"""

# Repo mastery priorities (using Legend in above string):
"""
For tracking progress in this legend scale, these are the levels needed:

- Tensor/data contracts (shapes, sizes, flow) -> 20% 🔸🔸🔸 
- Core math & algorithms (losses, diffusion schedule, EMA) -> 15% 🔸🔸
- Architecture interfaces (ResBlock, AttentionBlock, UNet forward signatures) -> 🔸🔸🔸 (or 🌟)
- Config schema & builder linkage -> 10% 🔸
- Training loop order op ops -> 10% 🔸🔸🔸
- Testing & invarients -> 10% 🔸🔸 
- External APIs (torch, wandb, tensorstore) -> 10% 🔸
- Logging/debug utilities -> 5% 🔺
- CLI/plumbing -> 3% 🔹
- Setup quirks -> 2% 🔹

"""


# Add one test per subsystem or block that:
# Asserts config roundtrips
# sanity-checks output shapes
# verifies NaN/inf-free gradients
# can be run on CPU + CI only
# Parameterize CPU and GPU runs, must work on both






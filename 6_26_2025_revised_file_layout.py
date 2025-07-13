
"""
latent_unet_v1/
├── run.py 🔹
├── README.md
├── pyproject.toml
├── .gitignore
├── .github/workflows/ci.yaml

├── configs/
│   ├── unet_config.yaml 🔹
│   ├── test_config.yaml 🔹
│   └── mock_data_config.yaml 🔹

├── model/
│   ├── config.py 🔹
│   ├── build_unet.py 🔹
│   └── unet.py  

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

├── diffusion/
│   ├── ddpm.py
│   ├── ddim.py
│   ├── edm.py
│   ├── sampler_registry.py
│   ├── sampler_utils.py
│   ├── forward_process.py
│   └── schedule.py

├── trainer/
│   ├── train_loop.py
│   ├── cluster_utils.py
│   ├── logger.py
│   └── losses.py

├── data/
│   ├── loader.py
│   ├── mock_dataset.py
│   └── dataset_registry.py

├── utils/
│   ├── debug.py
│   ├── visualizer.py
│   ├── memory_tools.py
│   ├── failure_injection.py
│   ├── tensor_inspect.py
│   ├── vanilla_checkpointing.py
│   │
│   ├── zarr_checkpointing/
│   │    ├── zarr_core.py
│   │    └── zarr_wrapper.py
│   │
│   └── tensorstore_checkpointing/
│       ├── __init__.py
│       ├── tensorstore_core.py
│       ├── tensorstore_wrapper.py
│       ├── schema_utils.py
│       ├── chunk_tuner.py
│       ├── registry.py
│       └── remote_utils.py

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
│   │   ├── test_unet.py 
│   │   ├── test_forward_process.py
│   │   ├── test_train_loop.py
│   │   └── test_ddpm_ddim_edm.py
│   │
│   ├── end_to_end/
│   │   ├── test_cuda_health.py
│   │   ├── test_training_pipeline.py
│   │   └── test_sampling_loop.py
│   │
│   ├── subsystems/                
│   │   ├── checkpointing/
│   │   │   ├── test_tensorstore.py
│   │   │   ├── test_tensorstore_schema_utils.py
│   │   │   ├── test_zarr.py
│   │   │   └── test_vanilla_checkpointing.py
│   │   ├── data/
│   │   │   └── test_data_pipeline.py 
│   │   ├── config/
|   |   |   ├── test_config_loading.py
│   │   │   ├── test_config_itegration.py
│   │   │   └── test_config_roundtrip.py  
│   │   │ 
│   │   ├── test_visualizer.py      
│   │   ├── test_debug_hooks.py 
│   │   └── test_failure_injection.py
│   │
│   ├── regression/
│   │   └── test_sequential_nan_bug.py  # (placeholder)
│   │
│   └── __init__.py

82 files
"""

# Add one test per subsystem or block that:
# Asserts config roundtrips
# sanity-checks output shapes
# verifies NaN/inf-free gradients
# can be run on CPU + CI only
# Parameterize CPU and GPU runs, must work on both






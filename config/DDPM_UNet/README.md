# Configuration Files

This directory contains YAML configuration files for training, validation, and inference.

## Usage

Run the main script with a config file and mode:

```bash
# Training
python main.py --config config/train_config.yaml --mode train

# Validation
python main.py --config config/val_config.yaml --mode val

# Inference
python main.py --config config/inference_config.yaml --mode inference
```

## Config File Structure

### General Structure

- `device`: Device to use (`auto`, `cuda`, `mps`, or `cpu`)
- `model`: Model configuration
  - `type`: Model type (e.g., `DDPM_UNet`)
  - `params`: Model-specific parameters
- `engine`: Engine configuration
  - `type`: Engine type (e.g., `DDPM_UNet`)
  - `params`: Engine-specific parameters
- `data`: Data loader configuration
  - `train`: Training data configuration (for train mode)
  - `val`: Validation data configuration (for train/val modes)
- `process`: Process-specific configuration
  - `train`: Training process parameters
  - `val`: Validation process parameters
  - `inference`: Inference process parameters

## Model Parameters (DDPM_UNet)

- `img_channels`: Number of image channels (default: 3)
- `base_channels`: Base number of channels (default: 64)
- `channel_mults`: Channel multipliers for each level (default: [1, 2, 4, 8])
- `num_res_blocks`: Number of residual blocks per level (default: 2)
- `time_emb_dim`: Time embedding dimension (default: 256)
- `num_classes`: Number of classes for conditional generation (null for unconditional)
- `dropout`: Dropout rate (default: 0.1)
- `attn_resolutions`: Which resolution levels to add attention (default: [1])
- `num_groups`: Number of groups for GroupNorm (default: 32)
- `init_pad`: Initial padding (default: 0)

## Engine Parameters (DDPM_UNet)

- `num_timesteps`: Number of diffusion timesteps (default: 1000)
- `beta_start`: Starting noise level (default: 0.0001)
- `beta_end`: Ending noise level (default: 0.02)
- `beta_schedule`: Noise schedule type (`linear` or `cosine`)

## Data Configuration

- `path`: Path to data directory
- `batch_size`: Batch size
- `shuffle`: Whether to shuffle data
- `num_workers`: Number of data loading workers
- `pin_memory`: Whether to pin memory
- `has_labels`: Whether dataset has labels (for conditional generation)

## Process Configuration

### Training

- `num_epochs`: Number of training epochs
- `save_dir`: Directory to save checkpoints
- `save_freq`: Frequency (in epochs) to save checkpoints
- `log_freq`: Frequency (in steps) to log training info
- `gradient_clip`: Gradient clipping value (null to disable)
- `resume_from`: Path to checkpoint to resume from (null to start fresh)
- `lr`: Learning rate
- `weight_decay`: Weight decay
- `betas`: Adam beta parameters
- `scheduler_type`: Scheduler type (`cosine`, `linear`, or `null`)
- `warmup_epochs`: Number of warmup epochs

### Validation

- `checkpoint_path`: Path to checkpoint to load
- `save_samples`: Whether to save generated samples
- `save_dir`: Directory to save validation results
- `num_samples_to_save`: Number of samples to save
- `metrics`: List of metric functions (currently not implemented)

### Inference

- `checkpoint_path`: Path to checkpoint to load
- `save_dir`: Directory to save inference results
- `num_samples`: Number of samples to generate
- `batch_size`: Batch size for generation
- `img_shape`: Image shape [C, H, W]
- `labels`: List of class indices for conditional generation (null for unconditional)
- `guidance_scale`: Classifier-free guidance scale
- `ddim`: Use DDIM sampling (faster)
- `ddim_steps`: Number of steps for DDIM sampling

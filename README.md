# Diffusion methods for remote sensing content generation

#### Author: Chanoch,  [Send a Email](kujou@foxmail.com)
#### Welcome :+1:_<big>`Fork and Star`</big>_:+1:, then we'll let you know when we update
#### -------------------------------------------------------------------------------------

A simple project for `text-to-image remote sensing image generation`.

## Todo List

##  Environment configuration

### Prerequisites

- Python >= 3.10
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. **Install uv** (if not already installed):
   Refer to the [official documentation](https://github.com/astral-sh/uv).

2. **Clone the repository**:
   ```bash
   git clone 
   cd rs-diffusion
   ```

3. **Install dependencies**:
   ```bash
   uv sync
   ```

4. **Activate the virtual environment**:
   ```bash
   # On macOS/Linux
   source .venv/bin/activate
   
   # On Windows
   .venv\Scripts\activate
   ```

## Usage

### Quick Start

The framework uses a unified `main.py` script with YAML configuration files.

#### Training

```bash
python main.py --config config/{model}/train_config.yaml --mode train
```

#### Validation

```bash
python main.py --config config/{model}/val_config.yaml --mode val
```

#### Inference

```bash
python main.py --config config/{model}/inference_config.yaml --mode inference
```

### Configuration Files

Configuration files are located in the `config/` directory. See `config/README.md` for detailed documentation.

Key configuration sections:
- **Model**: Model architecture parameters
- **Engine**: Model-specific engine parameters (e.g., diffusion timesteps)
- **Data**: Data loader configuration
- **Process**: Process-specific parameters (training, validation, inference)

### Distributed Training (Multi-GPU and Multi-Node)

#### Single Node Multi-GPU

Using `torchrun` (recommended):
```bash
torchrun --nproc_per_node=4 main.py --config config/DDPM_UNet/train_config.yaml --mode train
```

Or using the provided script:
```bash
./scripts/launch_distributed.sh config/DDPM_UNet/train_config.yaml train 4
```

#### Multi-Node Training

Using `torchrun`:
```bash
# On node 0
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=4 --master_addr="<node0_ip>" --master_port=29500 main.py --config config/DDPM_UNet/train_config.yaml --mode train

# On node 1
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=4 --master_addr="<node0_ip>" --master_port=29500 main.py --config config/DDPM_UNet/train_config.yaml --mode train
```

#### SLURM Cluster

For SLURM clusters, use the provided script:
```bash
sbatch scripts/launch_distributed_slurm.sh
```

#### Configuration

Distributed training is automatically detected via environment variables (`RANK`, `WORLD_SIZE`, `LOCAL_RANK`). The framework will:
- Automatically wrap the model with DDP
- Use `DistributedSampler` for data loading
- Only save checkpoints and log to TensorBoard on rank 0
- Synchronize losses across all processes

You can configure distributed training options in the config file:
```yaml
distributed:
  enabled: false  # Set to true to enable (or use environment variables)
  find_unused_parameters: false  # Set to true if model has unused parameters
```

**Note**: The effective batch size is `batch_size * num_gpus * num_nodes`. Adjust your learning rate accordingly.

### Custom Dataset

The framework uses `torchvision.datasets.ImageFolder` for ImageNet-like directory structures. Your dataset should be organized as:
```
data/
  train/
    class1/
      img1.jpg
      img2.jpg
    class2/
      img3.jpg
      ...
  val/
    class1/
      ...
```

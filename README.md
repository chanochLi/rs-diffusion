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

### Custom Dataset

The `ImageDataset` class in `main.py` is a placeholder. Replace it with your actual dataset implementation that:
- Loads images from your data source
- Optionally loads labels for conditional generation
- Returns dictionaries with `'images'` key (and optionally `'labels'` key)

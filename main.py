"""
Unified main script for training, validation, and inference.
Uses YAML config files to control all parameters.
"""
import argparse
import yaml
import sys
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from models import UNet
from engine.DDPM_UNet import DDPM_UNetEngine
from tools.train import TrainProcess
from tools.val import ValProcess
from tools.inference import InferenceProcess
from tools.distributed import init_distributed, wrap_model, is_main_process, get_distributed_sampler


def collate_fn(batch):
    """Collate function for DataLoader."""
    # ImageFolder returns (image, target) tuples
    images = torch.stack([item[0] for item in batch])
    result = {'images': images}
    
    # Check if labels are present (ImageFolder always provides labels)
    if len(batch[0]) > 1:
        labels = torch.tensor([item[1] for item in batch])
        result['labels'] = labels
    
    return result


def create_model(config: dict) -> torch.nn.Module:
    """
    Create model based on config.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Model instance
    """
    model_type = config.get('type', 'DDPM_UNet')
    
    if model_type == 'DDPM_UNet':
        model_config = config.get('params', {})
        return UNet(
            img_channels=model_config.get('img_channels', 3),
            base_channels=model_config.get('base_channels', 64),
            channel_mults=tuple(model_config.get('channel_mults', [1, 2, 4, 8])),
            num_res_blocks=model_config.get('num_res_blocks', 2),
            time_emb_dim=model_config.get('time_emb_dim', 256),
            num_classes=model_config.get('num_classes', None),
            act=model_config.get('act', 'relu'),
            dropout=model_config.get('dropout', 0.1),
            attn_resolutions=tuple(model_config.get('attn_resolutions', [])),
            num_groups=model_config.get('num_groups', 32),
            init_pad=model_config.get('init_pad', 0),
            num_heads=model_config.get('num_heads', 1),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_engine(config: dict, model: torch.nn.Module, device: torch.device) -> DDPM_UNetEngine:
    """
    Create engine based on config.
    
    Args:
        config: Engine configuration dictionary
        model: Model instance
        device: Device to run on
        
    Returns:
        Engine instance
    """
    engine_type = config.get('type', 'DDPM_UNet')
    
    if engine_type == 'DDPM_UNet':
        engine_config = config.get('params', {})
        return DDPM_UNetEngine(
            model=model,
            device=device,
            num_timesteps=engine_config.get('num_timesteps', 1000),
            beta_start=engine_config.get('beta_start', 0.0001),
            beta_end=engine_config.get('beta_end', 0.02),
            beta_schedule=engine_config.get('beta_schedule', 'linear'),
            config=config
        )
    else:
        raise ValueError(f"Unknown engine type: {engine_type}")


def create_data_loaders(config: dict) -> tuple[DataLoader | None, DataLoader | None]:
    """
    Create data loaders based on config.
    
    Args:
        config: Data configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    
    data_config = config.get('data', {})
    
    # Create transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_loader = None
    val_loader = None
    
    # Training data
    if 'train' in data_config:
        train_config = data_config['train']
        train_dataset = ImageFolder(
            root=train_config.get('path', './data/train'),
            transform=transform
        )
        
        # DDP的sampler
        train_sampler = get_distributed_sampler(
            train_dataset,
            shuffle=train_config.get('shuffle', True)
        )
        
        # Shuffle should be False when using DistributedSampler
        shuffle = train_config.get('shuffle', True) if train_sampler is None else False
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_config.get('batch_size', 16),
            shuffle=shuffle,
            sampler=train_sampler,
            num_workers=train_config.get('num_workers', 4),
            collate_fn=collate_fn,
            pin_memory=train_config.get('pin_memory', True)
        )
    
    # Validation data
    if 'val' in data_config:
        val_config = data_config['val']
        val_dataset = ImageFolder(
            root=val_config.get('path', './data/val'),
            transform=transform
        )
        
        # DDP的sampler
        val_sampler = get_distributed_sampler(
            val_dataset,
            shuffle=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_config.get('batch_size', 16),
            shuffle=False,
            sampler=val_sampler,
            num_workers=val_config.get('num_workers', 4),
            collate_fn=collate_fn,
            pin_memory=val_config.get('pin_memory', True)
        )
    
    return train_loader, val_loader


def get_device(config: dict, distributed: bool = False) -> torch.device:
    """
    Get device from config.
    
    Args:
        config: General configuration
        distributed: Whether distributed training is enabled
        
    Returns:
        torch.device
    """
    # 多卡
    if distributed and 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        if torch.cuda.is_available():
            return torch.device(f'cuda:{local_rank}')
    
    # 选择
    device_name = config.get('device', 'auto')
    if device_name == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    else:
        return torch.device(device_name)


def main():
    parser = argparse.ArgumentParser(description='Train, validate, or run inference on image generation models')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--mode', type=str, choices=['train', 'val', 'inference'], required=True, help='Mode to run: train, val, or inference')
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 多卡
    rank, world_size, distributed = init_distributed()
    
    # log
    if distributed:
        if is_main_process():
            print(f"Initialized distributed training: world_size={world_size}, rank={rank}")
    else:
        print("Running in single-process mode")
    
    # Get device
    device = get_device(config, distributed=distributed)
    if is_main_process():
        print(f"Using device: {device}")
    
    # Create model
    model = create_model(config.get('model', {}))
    if is_main_process():
        print(f"Model created: {config.get('model', {}).get('type', 'DDPM_UNet')}")
    
    # 多卡使用DDP封装模型
    model = wrap_model(model, device, find_unused_parameters=config.get('distributed', {}).get('find_unused_parameters', False))
    
    # 创建engine
    engine = create_engine(config.get('engine', {}), model, device)
    if is_main_process():
        print(f"Engine created: {config.get('engine', {}).get('type', 'DDPM_UNet')}")
    
    # 不同模式
    if args.mode == 'train':
        if is_main_process():
            print("\n=== Starting Training ===")
        train_loader, val_loader = create_data_loaders(config)
        
        assert train_loader is not None, "Training data loader not found in config"
        
        train_config = config.get('process', {}).get('train', {})
        train_process = TrainProcess(
            engine=engine,
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=train_config.get('save_dir', './checkpoints'),
            save_freq=train_config.get('save_freq', 10),
            log_freq=train_config.get('log_freq', 100),
            gradient_clip=train_config.get('gradient_clip', None),
            resume_from=train_config.get('resume_from', None),
            tensorboard_dir=train_config.get('tensorboard_dir', None),
            distributed=distributed,
            mixed_precision=train_config.get('mixed_precision', None),
            ema_decay=train_config.get('ema_decay', None),
            # Optimizer and scheduler params
            lr=train_config.get('lr', 1e-4),
            weight_decay=train_config.get('weight_decay', 0.0),
            betas=tuple(train_config.get('betas', [0.9, 0.999])),
            scheduler_type=train_config.get('scheduler_type', 'cosine'),
            num_epochs=train_config.get('num_epochs', 100),
            warmup_epochs=train_config.get('warmup_epochs', 0),
            accumulation_steps=train_config.get('accumulation_steps', 1),
        )
        train_process.run()
    
    elif args.mode == 'val':
        print("\n=== Starting Validation ===")
        _, val_loader = create_data_loaders(config)
        
        assert val_loader is not None, "Validation data loader not found in config"
        
        val_config = config.get('process', {}).get('val', {})
        val_process = ValProcess(
            engine=engine,
            val_loader=val_loader,
            metrics=val_config.get('metrics', None),
            save_samples=val_config.get('save_samples', False),
            save_dir=val_config.get('save_dir', './val_results'),
            num_samples_to_save=val_config.get('num_samples_to_save', 10),
        )
        
        # 加载模型
        checkpoint_path = val_config.get('checkpoint_path', None)
        if checkpoint_path:
            engine.load_checkpoint(checkpoint_path)
            print(f"Loaded checkpoint from {checkpoint_path}")
        
        val_process.run()
    
    elif args.mode == 'inference':
        print("\n=== Starting Inference ===")
        inference_config = config.get('process', {}).get('inference', {})
        
        inference_process = InferenceProcess(
            engine=engine,
            checkpoint_path=inference_config.get('checkpoint_path', None),
            save_dir=inference_config.get('save_dir', './inference_results'),
        )
        
        # 标签
        labels = inference_config.get('labels', None)
        if labels is not None:
            labels = torch.tensor(labels, device=device)
        
        # Generate samples
        samples = inference_process.run(
            num_samples=inference_config.get('num_samples', 1),
            batch_size=inference_config.get('batch_size', 1),
            img_shape=tuple(inference_config.get('img_shape', [3, 32, 32])),
            labels=labels,
            guidance_scale=inference_config.get('guidance_scale', 1.0),
            ddim=inference_config.get('ddim', False),
            ddim_steps=inference_config.get('ddim_steps', 50),
        )
        
        print(f"Generated {len(samples)} samples")


if __name__ == '__main__':
    main()

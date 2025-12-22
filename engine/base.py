"""
Base classes for engine framework.
Provides abstract base engine class that can be applied to all models.
"""
from abc import ABC, abstractmethod
from typing import Any
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class BaseEngine(ABC):
    """
    Abstract base class for model-specific engines.
    Each model should implement this class to define its specific behavior.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: dict[str, Any] | None = None
    ):
        """
        Initialize the engine.
        
        Args:
            model: The neural network model
            device: Device to run on (cuda, mps, cpu)
            config: Optional configuration dictionary
        """
        self.model = model.to(device)
        self.device = device
        self.config = config or {}
        
    @abstractmethod
    def forward_step(
        self,
        batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Perform a single forward pass.
        
        Args:
            batch: Dictionary containing input data (e.g., {'images': ..., 'labels': ...})
            
        Returns:
            Dictionary containing model outputs
        """
        pass
    
    @abstractmethod
    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute loss from model outputs and batch data.
        
        Args:
            outputs: Model outputs from forward_step
            batch: Original batch data
            
        Returns:
            Loss tensor (scalar)
        """
        pass
    
    @abstractmethod
    def sample(
        self,
        num_samples: int,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate samples from the model.
        
        Args:
            num_samples: Number of samples to generate
            **kwargs: Additional model-specific arguments
            
        Returns:
            Generated samples tensor
        """
        pass
    
    @abstractmethod
    def get_optimizer(self, **kwargs) -> Optimizer:
        """
        Create and return optimizer for training.
        
        Args:
            **kwargs: Optimizer-specific arguments (lr, weight_decay, etc.)
            
        Returns:
            Optimizer instance
        """
        pass
    
    def get_scheduler(
        self,
        optimizer: Optimizer,
        **kwargs
    ) -> _LRScheduler | None:
        """
        Create and return learning rate scheduler (optional).
        
        Args:
            optimizer: The optimizer to schedule
            **kwargs: Scheduler-specific arguments
            
        Returns:
            Scheduler instance or None
        """
        return None
    
    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        optimizer: Optimizer | None = None,
        scheduler: _LRScheduler | None = None,
        **kwargs
    ):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch number
            optimizer: Optional optimizer state
            scheduler: Optional scheduler state
            **kwargs: Additional items to save
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        checkpoint.update(kwargs)
        torch.save(checkpoint, path)
    
    def load_checkpoint(
        self,
        path: str,
        optimizer: Optimizer | None = None,
        scheduler: _LRScheduler | None = None
    ):
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint file
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 避免DDP在checkpoint中加入的module.前缀
        new_state = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            if k.startswith("module."):
                k = k[len("module."):]
            new_state[k] = v
        
        self.model.load_state_dict(new_state)
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint.get('epoch', 0)


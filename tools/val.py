"""
Validation process for image generation models.
"""
from collections.abc import Callable
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from .base import BaseProcess, BaseEngine


class ValProcess(BaseProcess):
    """
    General validation process that can be applied to any model.
    """
    
    def __init__(
        self,
        engine: BaseEngine,
        val_loader: DataLoader,
        metrics: list[Callable] | None = None,
        save_samples: bool = False,
        save_dir: str = "./val_results",
        num_samples_to_save: int = 10,
        **kwargs
    ):
        """
        Initialize validation process.
        
        Args:
            engine: Model-specific engine instance
            val_loader: Validation data loader
            metrics: Optional list of metric functions to compute
            save_samples: Whether to save generated samples
            save_dir: Directory to save validation results
            num_samples_to_save: Number of samples to save
            **kwargs: Additional arguments
        """
        super().__init__(engine, **kwargs)
        self.val_loader = val_loader
        self.metrics = metrics or []
        self.save_samples = save_samples
        self.save_dir = save_dir
        self.num_samples_to_save = num_samples_to_save
        
        if self.save_samples:
            os.makedirs(save_dir, exist_ok=True)
    
    def run(self) -> dict[str, float]:
        """
        Run the validation process.
        
        Returns:
            Dictionary containing validation metrics
        """
        self.engine.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        all_metrics = {f"metric_{i}": [] for i in range(len(self.metrics))}
        saved_count = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validating")
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                batch = self._move_to_device(batch)
                
                # Forward pass
                outputs = self.engine.forward_step(batch)
                loss = self.engine.compute_loss(outputs, batch)
                
                # Accumulate loss
                total_loss += loss.item()
                num_batches += 1
                
                # Compute metrics
                for i, metric_fn in enumerate(self.metrics):
                    metric_value = metric_fn(outputs, batch)
                    all_metrics[f"metric_{i}"].append(metric_value)
                
                # Save samples if requested
                if self.save_samples and saved_count < self.num_samples_to_save:
                    self._save_samples(batch, outputs, saved_count)
                    saved_count += 1
                
                # Update progress bar
                current_loss = total_loss / num_batches
                pbar.set_postfix({"loss": f"{current_loss:.4f}"})
        
        # Compute average metrics
        results = {
            "loss": total_loss / num_batches if num_batches > 0 else 0.0
        }
        
        for metric_name, metric_values in all_metrics.items():
            if metric_values:
                results[metric_name] = sum(metric_values) / len(metric_values)
        
        # Print results
        print("\nValidation Results:")
        for key, value in results.items():
            print(f"  {key}: {value:.4f}")
        
        return results
    
    def _move_to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Move batch tensors to the appropriate device.
        
        Args:
            batch: Batch dictionary
            
        Returns:
            Batch with tensors moved to device
        """
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.engine.device)
            else:
                device_batch[key] = value
        return device_batch
    
    def _save_samples(
        self,
        batch: dict[str, torch.Tensor],
        outputs: dict[str, torch.Tensor],
        sample_idx: int
    ):
        """
        Save generated samples to disk.
        
        Args:
            batch: Input batch
            outputs: Model outputs
            sample_idx: Index of the sample
        """
        # This is a placeholder - should be implemented based on specific model needs
        # For image generation, you might want to save images using torchvision
        pass

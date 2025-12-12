"""
Inference process for image generation models.
"""
import torch
import os
from tqdm import tqdm

from .base import BaseProcess, BaseEngine


class InferenceProcess(BaseProcess):
    """
    General inference process that can be applied to any model.
    """
    
    def __init__(
        self,
        engine: BaseEngine,
        checkpoint_path: str | None = None,
        save_dir: str = "./inference_results",
        **kwargs
    ):
        """
        Initialize inference process.
        
        Args:
            engine: Model-specific engine instance
            checkpoint_path: Optional path to checkpoint to load
            save_dir: Directory to save inference results
            **kwargs: Additional arguments
        """
        super().__init__(engine, **kwargs)
        self.save_dir = save_dir
        
        # Load checkpoint if provided
        if checkpoint_path is not None:
            self.engine.load_checkpoint(checkpoint_path)
            print(f"Loaded checkpoint from {checkpoint_path}")
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Set model to eval mode
        self.engine.model.eval()
    
    def run(
        self,
        num_samples: int = 1,
        batch_size: int = 1,
        **kwargs
    ) -> list[torch.Tensor]:
        """
        Run inference to generate samples.
        
        Args:
            num_samples: Number of samples to generate
            batch_size: Batch size for generation
            **kwargs: Additional model-specific arguments for sampling
            
        Returns:
            List of generated samples
        """
        all_samples = []
        
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        with torch.no_grad():
            pbar = tqdm(range(num_batches), desc="Generating samples")
            for batch_idx in pbar:
                # Determine how many samples in this batch
                current_batch_size = min(batch_size, num_samples - len(all_samples))
                
                # Generate samples
                samples_tensor = self.engine.sample(
                    num_samples=current_batch_size,
                    **kwargs
                )
                
                # Save samples (samples_tensor is [batch_size, C, H, W])
                for i in range(samples_tensor.shape[0]):
                    sample = samples_tensor[i]
                    sample_idx = len(all_samples)
                    self._save_sample(sample, sample_idx)
                    all_samples.append(sample)
                
                pbar.set_postfix({"generated": len(all_samples)})
        
        print(f"\nGenerated {len(all_samples)} samples")
        print(f"Results saved to {self.save_dir}")
        
        return all_samples
    
    def generate_from_batch(
        self,
        batch: dict[str, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """
        Generate samples from a given batch (e.g., conditional generation).
        
        Args:
            batch: Input batch dictionary
            **kwargs: Additional model-specific arguments
            
        Returns:
            Generated samples
        """
        # Move batch to device
        batch = self._move_to_device(batch)
        
        with torch.no_grad():
            outputs = self.engine.forward_step(batch)
            # Extract generated samples from outputs
            # This depends on the specific model implementation
            if "samples" in outputs:
                return outputs["samples"]
            elif "images" in outputs:
                return outputs["images"]
            else:
                # Fallback: return first tensor in outputs
                return list(outputs.values())[0]
    
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
    
    def _save_sample(self, sample: torch.Tensor, sample_idx: int):
        """
        Save a single sample to disk.
        
        Args:
            sample: Sample tensor to save
            sample_idx: Index of the sample
        """
        # This is a placeholder - should be implemented based on specific model needs
        # For image generation, you might want to save images using torchvision
        sample_path = os.path.join(self.save_dir, f"sample_{sample_idx:04d}.pt")
        torch.save(sample.cpu(), sample_path)

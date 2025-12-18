"""
DDPM_UNet specific engine implementation.
Handles the forward diffusion process, loss computation, and sampling for DDPM.
"""
from typing import Any
import torch
import torch.nn as nn
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import _LRScheduler
import math

from models import UNet
from .base import BaseEngine


class DDPM_UNetEngine(BaseEngine):
    """
    Engine for DDPM_UNet model.
    Implements DDPM-specific forward pass, loss computation, and sampling.
    """
    
    def __init__(
        self,
        model: UNet,
        device: torch.device,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        config: dict[str, Any] | None = None
    ):
        """
        Initialize DDPM engine.
        
        Args:
            model: UNet model instance
            device: Device to run on
            num_timesteps: Number of diffusion timesteps
            beta_start: Starting noise level
            beta_end: Ending noise level
            beta_schedule: Schedule type ("linear" or "cosine")
            config: Optional configuration dictionary
        """
        super().__init__(model, device, config)
        self.num_timesteps = num_timesteps
        
        # Setup noise schedule
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        elif beta_schedule == "cosine":
            # Cosine schedule as in Improved DDPM
            s = 0.008
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps, device=device)
            alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Precompute values for efficient sampling
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # For sampling
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
    
    def forward_step(
        self,
        batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Perform forward diffusion step.
        
        Args:
            batch: Dictionary containing 'images' and optionally 'labels'
            
        Returns:
            Dictionary with 'noise_pred' (predicted noise) and 'noise' (actual noise)
        """
        images = batch['images'].to(self.device)
        labels = batch.get('labels', None)
        if labels is not None:
            labels = labels.to(self.device)
        
        batch_size = images.shape[0]
        
        # Sample random timesteps
        t = torch.randint(
            0, self.num_timesteps, (batch_size,), device=self.device
        ).long()
        
        # Sample noise
        noise = torch.randn_like(images)
        
        # Add noise to images (forward diffusion)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(batch_size, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(batch_size, 1, 1, 1)
        noisy_images = sqrt_alphas_cumprod_t * images + sqrt_one_minus_alphas_cumprod_t * noise
        
        # Predict noise
        noise_pred = self.model(noisy_images, t=t.float(), label=labels)
        
        return {
            'noise_pred': noise_pred,
            'noise': noise,
            't': t
        }
    
    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute MSE loss between predicted and actual noise.
        
        Args:
            outputs: Outputs from forward_step
            batch: Original batch (not used here, but kept for interface consistency)
            
        Returns:
            Loss tensor
        """
        noise_pred = outputs['noise_pred']
        noise = outputs['noise']
        
        # Simple MSE loss
        loss = nn.functional.mse_loss(noise_pred, noise)
        
        return loss
    
    def sample(
        self,
        num_samples: int,
        img_shape: tuple = (3, 32, 32),
        labels: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
        ddim: bool = False,
        ddim_steps: int = 50
    ) -> torch.Tensor:
        """
        Generate samples using reverse diffusion process.
        
        Args:
            num_samples: Number of samples to generate
            img_shape: Shape of images (C, H, W)
            labels: Optional class labels for conditional generation
            guidance_scale: Classifier-free guidance scale (if > 1.0)
            ddim: Whether to use DDIM sampling (faster)
            ddim_steps: Number of steps for DDIM sampling
            
        Returns:
            Generated samples tensor [num_samples, C, H, W]
        """
        self.model.eval()
        
        # Start from pure noise
        samples = torch.randn(
            (num_samples, *img_shape),
            device=self.device
        )
        
        if ddim:
            return self._ddim_sample(samples, labels, ddim_steps, guidance_scale)
        else:
            return self._ddpm_sample(samples, labels, guidance_scale)
    
    def _ddpm_sample(
        self,
        samples: torch.Tensor,
        labels: torch.Tensor | None,
        guidance_scale: float
    ) -> torch.Tensor:
        """
        Standard DDPM sampling (full reverse process).
        """
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((samples.shape[0],), i, device=self.device, dtype=torch.long)
            
            # Predict noise
            with torch.no_grad():
                if guidance_scale > 1.0 and labels is not None:
                    # Classifier-free guidance
                    noise_pred_uncond = self.model(samples, t=t.float(), label=None)
                    noise_pred_cond = self.model(samples, t=t.float(), label=labels)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = self.model(samples, t=t.float(), label=labels)
            
            # Compute coefficients
            alpha_t = self.alphas[t].view(-1, 1, 1, 1)
            alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
            beta_t = self.betas[t].view(-1, 1, 1, 1)
            
            if i > 0:
                # Compute predicted x_0
                pred_x0 = (samples - torch.sqrt(1.0 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
                
                # Sample from posterior
                posterior_mean = (
                    torch.sqrt(alpha_cumprod_t) * beta_t / (1.0 - alpha_cumprod_t) * pred_x0 +
                    torch.sqrt(alpha_t) * (1.0 - self.alphas_cumprod_prev[t]) / (1.0 - alpha_cumprod_t) * samples
                )
                posterior_variance = self.posterior_variance[t].view(-1, 1, 1, 1)
                
                noise = torch.randn_like(samples) if i > 0 else torch.zeros_like(samples)
                samples = posterior_mean + torch.sqrt(posterior_variance) * noise
            else:
                # Last step: no noise
                pred_x0 = (samples - torch.sqrt(1.0 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
                samples = pred_x0
        
        return samples
    
    def _ddim_sample(
        self,
        samples: torch.Tensor,
        labels: torch.Tensor | None,
        ddim_steps: int,
        guidance_scale: float
    ) -> torch.Tensor:
        """
        DDIM sampling (deterministic, faster).
        """
        # Select timesteps for DDIM
        step_size = self.num_timesteps // ddim_steps
        timesteps = list(range(0, self.num_timesteps, step_size))
        timesteps = timesteps[::-1]  # Reverse
        
        for i, t_idx in enumerate(timesteps):
            t = torch.full((samples.shape[0],), t_idx, device=self.device, dtype=torch.long)
            
            # Predict noise
            with torch.no_grad():
                if guidance_scale > 1.0 and labels is not None:
                    noise_pred_uncond = self.model(samples, t=t.float(), label=None)
                    noise_pred_cond = self.model(samples, t=t.float(), label=labels)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = self.model(samples, t=t.float(), label=labels)
            
            # Compute predicted x_0
            alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
            pred_x0 = (samples - torch.sqrt(1.0 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
            
            # Compute direction pointing to x_t
            dir_xt = torch.sqrt(1.0 - alpha_cumprod_t) * noise_pred
            
            if i < len(timesteps) - 1:
                # Compute next timestep
                next_t_idx = timesteps[i + 1]
                alpha_cumprod_next = self.alphas_cumprod[next_t_idx].view(-1, 1, 1, 1)
                samples = torch.sqrt(alpha_cumprod_next) * pred_x0 + torch.sqrt(1.0 - alpha_cumprod_next) * dir_xt
            else:
                # Last step
                samples = pred_x0
        
        return samples
    
    def get_optimizer(
        self,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        betas: tuple = (0.9, 0.999),
        **kwargs
    ) -> Optimizer:
        """
        Create Adam optimizer for training.
        
        Args:
            lr: Learning rate
            weight_decay: Weight decay
            betas: Adam beta parameters
            **kwargs: Additional optimizer arguments
            
        Returns:
            Adam optimizer
        """
        return Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            **kwargs
        )
    
    def get_scheduler(
        self,
        optimizer: Optimizer,
        scheduler_type: str = "cosine",
        num_epochs: int = 100,
        warmup_epochs: int = 0,
        **kwargs
    ) -> _LRScheduler | None:
        """
        Create learning rate scheduler.
        
        Args:
            optimizer: Optimizer instance
            scheduler_type: Type of scheduler ("cosine", "linear", or None)
            num_epochs: Total number of epochs
            warmup_epochs: Number of warmup epochs
            **kwargs: Additional scheduler arguments
            
        Returns:
            Scheduler instance or None
        """
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, ConstantLR
        
        if scheduler_type is None:
            return None
        
        # Base scheduler
        if scheduler_type == "cosine":
            base_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, **kwargs)
        elif scheduler_type == "linear":
            base_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=num_epochs - warmup_epochs, **kwargs)
        else:
            return None
        
        # Add warmup if needed
        if warmup_epochs > 0:
            warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
            return SequentialLR(optimizer, schedulers=[warmup_scheduler, base_scheduler], milestones=[warmup_epochs])   # type: ignore
        else:
            return base_scheduler   # type: ignore

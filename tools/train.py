"""
Training process for image generation models.
"""
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from typing import Optional

from .base import BaseProcess, BaseEngine
from .distributed import is_main_process, all_reduce_mean, barrier


class TrainProcess(BaseProcess):
    """
    General training process that can be applied to any model.
    """
    
    def __init__(
        self,
        engine: BaseEngine,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        num_epochs: int = 100,
        save_dir: str = "./checkpoints",
        save_freq: int = 10,
        log_freq: int = 100,
        gradient_clip: float | None = None,
        resume_from: str | None = None,
        tensorboard_dir: Optional[str] = None,
        distributed: bool = False,
        **kwargs
    ):
        """
        Initialize training process.
        
        Args:
            engine: Model-specific engine instance
            train_loader: Training data loader
            val_loader: Optional validation data loader
            num_epochs: Number of training epochs
            save_dir: Directory to save checkpoints
            save_freq: Frequency (in epochs) to save checkpoints
            log_freq: Frequency (in steps) to log training info
            gradient_clip: Optional gradient clipping value
            resume_from: Optional path to checkpoint to resume from
            tensorboard_dir: Optional directory for TensorBoard logs
            distributed: Whether distributed training is enabled
            **kwargs: Additional arguments passed to optimizer/scheduler
        """
        super().__init__(engine, **kwargs)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.log_freq = log_freq
        self.gradient_clip = gradient_clip
        self.distributed = distributed
        
        # Initialize TensorBoard writer (only on main process)
        self.writer = None
        if tensorboard_dir is not None and is_main_process():
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tensorboard_dir)
            print(f"TensorBoard logging enabled. Logs will be saved to: {tensorboard_dir}")
        
        # Separate optimizer and scheduler kwargs
        scheduler_kwargs = {
            'scheduler_type': kwargs.pop('scheduler_type', 'cosine'),
            'num_epochs': num_epochs,  # Use the parameter, not from kwargs
            'warmup_epochs': kwargs.pop('warmup_epochs', 0),
        }
        # Remaining kwargs are for optimizer
        optimizer_kwargs = kwargs
        
        # Create optimizer and scheduler
        self.optimizer = engine.get_optimizer(**optimizer_kwargs)
        self.scheduler = engine.get_scheduler(self.optimizer, **scheduler_kwargs)
        
        # Create save directory (only on main process)
        if is_main_process():
            os.makedirs(save_dir, exist_ok=True)
        
        # Resume from checkpoint if provided
        self.start_epoch = 0
        if resume_from is not None:
            self.start_epoch = engine.load_checkpoint(
                resume_from,
                optimizer=self.optimizer,
                scheduler=self.scheduler
            )
            if is_main_process():
                print(f"Resumed training from epoch {self.start_epoch}")
            # Synchronize all processes after loading checkpoint
            if distributed:
                barrier()
    
    def run(self):
        """
        Run the training loop.
        """
        self.engine.model.train()
        
        # Set epoch for DistributedSampler if using distributed training
        if self.distributed and hasattr(self.train_loader.sampler, 'set_epoch'):
            sampler = self.train_loader.sampler
        
        for epoch in range(self.start_epoch, self.num_epochs):
            # Set epoch for DistributedSampler
            if self.distributed and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            
            # Training phase
            train_loss = self._train_epoch(epoch)
            
            # Validation phase (if validation loader provided)
            val_loss = None
            if self.val_loader is not None:
                if self.distributed and hasattr(self.val_loader.sampler, 'set_epoch'):
                    self.val_loader.sampler.set_epoch(epoch)
                val_loss = self._validate_epoch()
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Logging (only on main process)
            if is_main_process():
                log_msg = f"Epoch [{epoch+1}/{self.num_epochs}] - Train Loss: {train_loss:.4f}"
                if val_loss is not None:
                    log_msg += f" - Val Loss: {val_loss:.4f}"
                print(log_msg)
                
                # Log to TensorBoard
                if self.writer is not None:
                    self.writer.add_scalar('Loss/Train', train_loss, epoch + 1)
                    if val_loss is not None:
                        self.writer.add_scalar('Loss/Validation', val_loss, epoch + 1)
                    if self.scheduler is not None:
                        current_lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.optimizer.param_groups[0]['lr']
                        self.writer.add_scalar('Learning_Rate', current_lr, epoch + 1)
            
            # Save checkpoint (only on main process)
            if is_main_process() and ((epoch + 1) % self.save_freq == 0 or epoch == self.num_epochs - 1):
                checkpoint_path = os.path.join(
                    self.save_dir,
                    f"checkpoint_epoch_{epoch+1}.pt"
                )
                self.engine.save_checkpoint(
                    checkpoint_path,
                    epoch=epoch + 1,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler
                )
                print(f"Saved checkpoint to {checkpoint_path}")
            
            # Synchronize all processes after each epoch
            if self.distributed:
                barrier()
        
        # Close TensorBoard writer (only on main process)
        if is_main_process() and self.writer is not None:
            self.writer.close()
            print("TensorBoard writer closed.")
    
    def _train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average training loss
        """
        total_loss = 0.0
        num_batches = 0
        
        # Only show progress bar on main process
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.num_epochs}",
            leave=False,
            disable=not is_main_process()
        )
        
        for step, batch in enumerate(pbar):
            # Move batch to device
            batch = self._move_to_device(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.engine.forward_step(batch)
            loss = self.engine.compute_loss(outputs, batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.engine.model.parameters(),
                    self.gradient_clip
                )
            
            self.optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1
            
            # Calculate global step for TensorBoard
            global_step = epoch * len(self.train_loader) + step + 1
            
            # Logging
            if (step + 1) % self.log_freq == 0:
                current_loss = total_loss / num_batches
                pbar.set_postfix({"loss": f"{current_loss:.4f}"})
                
                # Log to TensorBoard (only on main process)
                if is_main_process() and self.writer is not None:
                    self.writer.add_scalar('Loss/Train_Step', current_loss, global_step)
                    if self.scheduler is not None:
                        current_lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.optimizer.param_groups[0]['lr']
                        self.writer.add_scalar('Learning_Rate_Step', current_lr, global_step)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Reduce loss across all processes in distributed training
        if self.distributed:
            avg_loss_tensor = torch.tensor(avg_loss, device=self.engine.device)
            avg_loss_tensor = all_reduce_mean(avg_loss_tensor)
            avg_loss = avg_loss_tensor.item()
        
        return avg_loss
    
    def _validate_epoch(self) -> float:
        """
        Validate for one epoch.
        
        Returns:
            Average validation loss
        """
        self.engine.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._move_to_device(batch)
                outputs = self.engine.forward_step(batch)
                loss = self.engine.compute_loss(outputs, batch)
                total_loss += loss.item()
                num_batches += 1
        
        self.engine.model.train()
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Reduce loss across all processes in distributed training
        if self.distributed:
            avg_loss_tensor = torch.tensor(avg_loss, device=self.engine.device)
            avg_loss_tensor = all_reduce_mean(avg_loss_tensor)
            avg_loss = avg_loss_tensor.item()
        
        return avg_loss
    
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

"""
Training process for image generation models.
"""
import os
import copy
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch.amp import autocast, GradScaler

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
        accumulation_steps: int = 1,
        resume_from: str | None = None,
        tensorboard_dir: str | None = None,
        distributed: bool = False,
        mixed_precision: str | None = None,
        ema_decay: float | None = None,
        **kwargs
    ):
        """
        Initialize training process.
        
        Args:
            engine: 具体方法的eigine对象
            train_loader: 训练dataloader
            val_loader: 验证dataloader
            num_epochs: epochs
            save_dir: ckpt保存目录
            save_freq: 保存ckpt的频率(epoch)
            log_freq: 记录日志的频率(step)
            gradient_clip: clip的阈值
            accumulation_steps: 梯度累积步数, 默认为1, 不累积
            resume_from: 恢复路径
            tensorboard_dir: tensorboard目录
            distributed: 是否多卡
            mixed_precision: 混合精度类型, 可选值: None (禁用), 'fp16', 'bf16'
            ema_decay: EMA衰减率, None表示禁用EMA
            **kwargs: 其余optimizer和scheduler的参数
        """
        super().__init__(engine, **kwargs)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.log_freq = log_freq
        self.gradient_clip = gradient_clip
        self.accumulation_steps = accumulation_steps
        self.distributed = distributed
        
        # 混合精度设置
        self.mixed_precision = mixed_precision
        self.scaler = None
        
        # 设置autocast的dtype
        if mixed_precision is not None and mixed_precision.lower() in ['fp16', 'bf16']:
            if mixed_precision.lower() == 'fp16':
                self.amp_dtype = torch.float16
                if self.engine.device.type == 'cuda':
                    self.scaler = GradScaler()
            elif mixed_precision.lower() == 'bf16':
                self.amp_dtype = torch.bfloat16
        else:
            self.amp_dtype = None
        
        # 初始化tensorboard
        self.writer = None
        if tensorboard_dir is not None and is_main_process():
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tensorboard_dir)
            print(f"TensorBoard logging enabled. Logs will be saved to: {tensorboard_dir}")
        
        # 抽取出给scheduler的参数
        scheduler_kwargs = {
            'scheduler_type': kwargs.pop('scheduler_type', 'cosine'),
            'num_epochs': num_epochs,
            'warmup_epochs': kwargs.pop('warmup_epochs', 0),
        }
        # 给optimizer的参数
        optimizer_kwargs = kwargs
        
        # 创建optimizer和lr scheduler
        self.optimizer = engine.get_optimizer(**optimizer_kwargs)
        self.scheduler = engine.get_scheduler(self.optimizer, **scheduler_kwargs)
        
        # 初始化EMA模型
        self.ema_decay = ema_decay
        self.ema_model = None
        if ema_decay is not None and ema_decay > 0:
            # DDP需要获取其中封装的module
            original_model = engine.model.module if hasattr(engine.model, 'module') else engine.model

            # 创建EMA模型
            self.ema_model = copy.deepcopy(original_model)
            self.ema_model.to(engine.device)
            self.ema_model.eval()  # EMA模型始终处于eval模式

            # 确保EMA模型参数不需要梯度
            for param in self.ema_model.parameters():
                param.requires_grad = False
                
            if is_main_process():
                print(f"EMA enabled with decay={ema_decay}")
        
        # 模型保存文件
        if is_main_process():
            os.makedirs(save_dir, exist_ok=True)
        
        # Resume
        self.start_epoch = 0
        if resume_from is not None:
            checkpoint = torch.load(resume_from, map_location=self.engine.device)
            self.start_epoch = engine.load_checkpoint(
                resume_from,
                optimizer=self.optimizer,
                scheduler=self.scheduler
            )
            
            # 恢复GradScaler状态
            if self.scaler is not None and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            # 恢复EMA模型状态
            if self.ema_model is not None and 'ema_model_state_dict' in checkpoint and checkpoint['ema_model_state_dict'] is not None:
                self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
                if is_main_process():
                    print("Restored EMA model state from checkpoint")
            
            if is_main_process():
                print(f"Resumed training from epoch {self.start_epoch}")
                
            # 同步所有卡
            if distributed:
                barrier()
    
    def _update_ema(self):
        """
        更新EMA模型的参数。
        EMA参数 = decay * EMA参数 + (1 - decay) * 模型参数
        """
        if self.ema_model is None:
            return
        
        # DDP需要获取其中封装的module
        model = self.engine.model.module if hasattr(self.engine.model, 'module') else self.engine.model
        
        # 更新EMA模型参数
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(model_param.data, alpha=1.0 - self.ema_decay)
    
    def run(self):
        """
        Run the training loop.
        """
        self.engine.model.train()
        
        for epoch in range(self.start_epoch, self.num_epochs):
            # 为分布式data loader设置epoch
            if self.distributed and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)  # type: ignore
            
            # 训练
            train_loss = self._train_epoch(epoch)
            
            # Validation phase (if validation loader provided)
            val_loss = None
            if self.val_loader is not None:
                if self.distributed and hasattr(self.val_loader.sampler, 'set_epoch'):
                    self.val_loader.sampler.set_epoch(epoch)  # type: ignore
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

                # 保存GradScaler状态
                scaler_state = self.scaler.state_dict() if self.scaler is not None else None
                
                # 保存EMA模型状态
                ema_state = self.ema_model.state_dict() if self.ema_model is not None else None
                    
                self.engine.save_checkpoint(
                    checkpoint_path,
                    epoch=epoch + 1,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler_state_dict=scaler_state,
                    ema_model_state_dict=ema_state
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
        
        # 显示当前进度
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.num_epochs}",
            leave=False,
            disable=not is_main_process()
        )
        
        #  在epoch开始时清零梯度
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(pbar):
            batch = self._move_to_device(batch)
            
            with autocast(self.engine.device.type, dtype=self.amp_dtype):
                outputs = self.engine.forward_step(batch)
                loss = self.engine.compute_loss(outputs, batch)
                # 将loss除以累积步数，以保持梯度尺度一致
                loss = loss / self.accumulation_steps
            
            # 反向传播
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            total_loss += loss.item() * self.accumulation_steps
            num_batches += 1
            
            # accumulation_steps步参数更新
            if (step + 1) % self.accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)

                # 梯度裁剪，防止训练不稳定
                if self.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.engine.model.parameters(), self.gradient_clip)
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                # 更新EMA模型
                self._update_ema()
                
                self.optimizer.zero_grad()
            
            global_step = epoch * len(self.train_loader) + step + 1
            
            # logging
            if (step + 1) % self.log_freq == 0:
                current_loss = total_loss / num_batches
                pbar.set_postfix({"loss": f"{current_loss:.4f}"})
                
                # tensorboard
                if is_main_process() and self.writer is not None:
                    self.writer.add_scalar('Loss/Train_Step', current_loss, global_step)
                    if self.scheduler is not None:
                        current_lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.optimizer.param_groups[0]['lr']
                        self.writer.add_scalar('Learning_Rate_Step', current_lr, global_step)
        
        # 处理最后一个不完整的累积批次
        if num_batches % self.accumulation_steps != 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)

            # 梯度裁剪，防止训练不稳定
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.engine.model.parameters(), self.gradient_clip)
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            # 更新EMA模型
            self._update_ema()
            
            self.optimizer.zero_grad()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # 分布式all reduce
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
        
        assert self.val_loader is not None, 'Should not run valid epoch when val_loader is None'
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._move_to_device(batch)
                # 统一使用autocast（验证时不需要梯度缩放）
                with autocast(self.engine.device.type, dtype=self.amp_dtype):
                    outputs = self.engine.forward_step(batch)
                    loss = self.engine.compute_loss(outputs, batch)
                total_loss += loss.item()
                num_batches += 1
        
        self.engine.model.train()
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # 分布式all reduce
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

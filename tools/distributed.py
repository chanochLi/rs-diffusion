"""
Distributed training utilities for multi-GPU and multi-node training.
"""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def init_distributed(
    backend: str = "nccl",
    init_method: str = "env://",
    world_size: int | None = None,
    rank: int | None = None
) -> tuple[int, int, bool]:
    """
    初始化多卡训练
    
    Args:
        backend: Distributed backend ('nccl' for GPU, 'gloo' for CPU)
        init_method: Initialization method ('env://' uses environment variables)
        world_size: Total number of processes (if None, uses env var)
        rank: Process rank (if None, uses env var)
        
    Returns:
        Tuple of (rank, world_size, is_distributed)
    """
    # Check if distributed training is enabled
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    elif 'SLURM_PROCID' in os.environ:
        # SLURM environment
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ.get('SLURM_NTASKS', 1))
        local_rank = int(os.environ.get('SLURM_LOCALID', 0))
    else:
        # Not in distributed mode
        return 0, 1, False
    
    # Set device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = 'nccl'
    else:
        backend = 'gloo'
    
    # Initialize process group
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank
    )
    
    # Set device for this process
    os.environ['LOCAL_RANK'] = str(local_rank)
    
    return rank, world_size, True


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_model(model: torch.nn.Module, device: torch.device, find_unused_parameters: bool = False) -> torch.nn.Module:
    """
    Wrap model with DistributedDataParallel if distributed training is enabled.
    
    Args:
        model: Model to wrap
        device: Device to use
        find_unused_parameters: Whether to find unused parameters in DDP
        
    Returns:
        Wrapped model (DDP if distributed, otherwise original model)
    """
    if dist.is_initialized():
        model = model.to(device)
        model = DDP(model, device_ids=[device.index] if device.type == 'cuda' else None, find_unused_parameters=find_unused_parameters)
        return model
    return model


def get_distributed_sampler(dataset, shuffle: bool = True):
    """
    Get DistributedSampler if distributed training is enabled.
    
    Args:
        dataset: Dataset to sample from
        shuffle: Whether to shuffle the data
        
    Returns:
        DistributedSampler if distributed, None otherwise
    """
    if dist.is_initialized():
        return DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=shuffle
        )
    return None


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    """Get the rank of the current process."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get the world size (total number of processes)."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reduce tensor across all processes and compute mean.
    
    Args:
        tensor: Tensor to reduce
        
    Returns:
        Reduced tensor (mean across all processes)
    """
    if not dist.is_initialized():
        return tensor
    
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def barrier():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()

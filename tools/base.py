"""
Base classes for training framework.
Provides abstract base process classes that can be applied to all engines.
"""
from abc import ABC, abstractmethod

from engine.base import BaseEngine

class BaseProcess(ABC):
    """
    Base class for training/validation/inference processes.
    """
    
    def __init__(
        self,
        engine: BaseEngine,
        **kwargs
    ):
        """
        Initialize the process.
        
        Args:
            engine: Model-specific engine instance
            **kwargs: Additional process-specific arguments
        """
        self.engine = engine
        self.kwargs = kwargs
    
    @abstractmethod
    def run(self, *args, **kwargs):
        """
        Run the process.
        """
        pass

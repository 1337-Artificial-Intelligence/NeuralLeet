from abc import ABC, abstractmethod
import numpy as np

class Neural:
    """Base class for all neural network modules."""
    def __init__(self) -> None:
        pass

class Layer(Neural, ABC):
    """Abstract base class for all neural network layers."""
    
    def __init__(self) -> None:
        super().__init__()
        
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        pass
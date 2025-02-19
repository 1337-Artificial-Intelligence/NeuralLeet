from abc import ABC, abstractmethod
import numpy as np

class Neuron:
    """Base class for neuron network components."""
    
    def __init__(self) -> None:
        """Initialize a new Neuron instance."""
        pass

class Layer(Neuron, ABC):
    """Abstract base class for layers."""

    def __init__(self) -> None:
        """Initialize a new Layer instance."""
        super().__init__()
        pass

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        pass

class NeuralNetwork(Layer, ABC):
    """Abstract base class for neural network models."""
    
    def __init__(self) -> None:
        """Initialize a new NeuralNetwork instance."""
        super().__init__()
        pass
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
    '''
        Abstract base class for neural network models.
        Parameters:
            - num_layers: int
                The number of layers in the neural network.
            - input_size: int
                The size of the input layer.
            - output_size: int
                The size of the output layer.
            - learning_rate: float
                The learning rate of the neural network.
            - batch_size: int
                The batch size of the neural network.
        Returns:
            None
    '''

    def __init__(
        self, num_layers: int, input_size: int, output_size: int,
            learning_rate: float = 0.01, batch_size: int = 128) -> None:

        super().__init__()
        self.number_of_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size

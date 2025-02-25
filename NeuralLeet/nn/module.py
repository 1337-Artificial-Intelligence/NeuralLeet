from abc import ABC, abstractmethod
import numpy as np


class Neuron:

    def __init__(self, num_inputs: int):
        self.num_inputs = num_inputs
        self.weights = np.random.randn(num_inputs)
        self.bias = 0


class Layer(ABC):
    '''
        Layer class for neural network models.
        Parameters:
            - num_neurons: int
                The number of neurons in the layer.
            - num_inputs: int
                The number of inputs to the each neuron in the current layer.
        Returns:
            None
    '''

    def __init__(self, num_neurons: int, num_inputs: int) -> None:
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        self.neurons = [Neuron(self.num_inputs) for _ in range(num_neurons)]

    @abstractmethod
    def forward(self, x_batch: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        pass


class NeuralNetwork(ABC):
    '''
        Abstract base class for neural network models.
        Parameters:
            - num_hidden_layers: int
                The number of layers in the neural network.
            - num_neurons_hidden: int
                The number of neurons in each hidden layer.
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
        self, num_hidden_layers: int, num_neurons_hidden: int,
        input_size: int, output_size: int,
            learning_rate: float, batch_size: int, epochs: int,
            layer_type: Layer = Layer
    ) -> None:

        self.num_of_hidden_layers = num_hidden_layers
        self.num_neurons_hidden = num_neurons_hidden
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.layers = []
        for i in range(num_hidden_layers):
            if i == 0:
                self.layers.append(layer_type(num_neurons_hidden, input_size))
            else:
                self.layers.append(
                    layer_type(num_neurons_hidden, num_neurons_hidden))
        self.layers.append(layer_type(output_size, num_neurons_hidden))

    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        '''
            Train the neural network model.
            Parameters:
                - x: np.ndarray
                    The input data.
                - y: np.ndarray
                    The target data.
            Returns:
                None
        '''

        # todo add debugging print statements for ech epoch
        for epoch in range(self.epochs):
            for i in range(0, len(x), self.batch_size):
                x_batch = x[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]
                for layer in self.layers:
                    x_batch = layer.forward(x_batch)

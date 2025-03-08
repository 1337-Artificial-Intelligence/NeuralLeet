from abc import ABC, abstractmethod
import numpy as np

from ..core.functional import activation_functions


class Layer(ABC):
    '''
        Layer class for neural network models.
        Parameters:
            - num_neurons: int
                The number of neurons in the layer.
            - num_inputs: int
                The number of inputs to the each neuron in the current layer.
            - activation_function: callable
                The activation function to use for the layer
            - activation_function_derivative: callable
                The derivative of the activation function
            - learning_rate: float
                The learning rate of the neural network.
            - is_output_layer: bool
                A flag to indicate if the layer is the output
        Returns:
            None
    '''

    def __init__(self, num_neurons: int, num_inputs: int, activation_function: callable,
                 activation_function_derivative: callable, learning_rate: float,
                 is_output_layer: bool = False
                 ) -> None:
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        self.weights = np.random.randn(num_neurons, num_inputs) * 0.01
        self.bias = np.zeros((num_neurons, 1))
        self.activation_function = activation_function
        self.is_output_layer = is_output_layer
        self.activation_function_derivative = activation_function_derivative
        self.learning_rate = learning_rate

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
            - epochs: int
                The number of epochs to train the neural network.
            - layer_type: Layer
                The type of layer to use in the neural network it should inherit from the Layer class.
            - h_activation: str
                The activation function for the hidden layers.
            - o_activation: str
                The activation function for the output layer.
            - loss: callable
                The loss function to use for training the neural network.
        Returns:
            None
    '''

    def __init__(
        self, num_hidden_layers: int, num_neurons_hidden: int,
        input_size: int, output_size: int,
        learning_rate: float, batch_size: int, epochs: int,
        layer_type: Layer, h_activation: str,
        o_activation: str, loss: callable
    ) -> None:
        self.num_of_hidden_layers = num_hidden_layers
        self.num_neurons_hidden = num_neurons_hidden
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.layers = []
        self.h_activation, self.h_activation_derivative = activation_functions[h_activation]
        self.o_activation, self.o_activation_derivative = activation_functions[o_activation]
        self.loss = loss

        for i in range(num_hidden_layers):
            if i == 0:
                self.layers.append(layer_type(
                    num_neurons_hidden, input_size, self.h_activation, self.h_activation_derivative, learning_rate))
            else:
                self.layers.append(layer_type(
                    num_neurons_hidden, num_neurons_hidden, self.h_activation, self.h_activation_derivative, learning_rate))
        self.layers.append(layer_type(
            output_size, num_neurons_hidden, self.o_activation, self.o_activation_derivative, learning_rate, is_output_layer=True))

    @abstractmethod
    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        ''''
            Train the neural network model.
            Parameters:
                - x: np.ndarray
                    The input data to the neural network.
                - y: np.ndarray
                    The target values for the input data.
            Returns:
                None
        '''
        pass
    
    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        '''
            Predict the output of the neural network model.
            Parameters:
                - x: np.ndarray
                    The input data to the neural network.
            Returns:
                - np.ndarray
                    The output of the neural network model.
        '''
        pass

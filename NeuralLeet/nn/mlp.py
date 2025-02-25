from NeuralLeet.nn.module import Layer, NeuralNetwork
import numpy as np


class MlpLayer(Layer):
    '''
    Multi-layer perceptron layer class, inherits from the Layer class.
    Parameters:
        - num_neurons: int
            The number of neurons in the layer.
        - num_inputs: int
            The number of inputs to the each neuron in the current layer.

    '''

    def __init__(self, num_neurons, num_inputs):
        super().__init__(num_neurons, num_inputs)

    # * Override the forward method
    def forward(self, x_batch):
        '''
        Forward pass for the layer.
        Parameters:
            - x_batch: np.ndarray
                The input batch of data.
        Returns:
            - np.ndarray
                The output of the layer
        '''
        results = np.zeros((x_batch.shape[0], self.num_neurons))
        for i in range(x_batch.shape[0]):
            predictions_i = [
                np.dot(neuron.weights, x_batch[i]) + neuron.bias
                for neuron in self.neurons]
            results[i] = predictions_i
        return results

    def backward(self, grad_output):
        pass


class Mlp(NeuralNetwork):
    '''
    Multi-layer perceptron class, inherits from the NeuralNetwork class.
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
        - h_activation: callable
            The activation function for the hidden layers.
        - o_activation: callable
            The activation function for the output layer
    '''

    def __init__(
            self, num_hidden_layers: int,
            num_neurons_hidden: int, input_size: int,
            h_activation: callable, o_activation: callable,
            output_size: int, learning_rate: float = 0.01,
            batch_size: int = 128, epochs: int = 100,
    ) -> None:

        # * Initialize the parent class NeuralNetwork
        super(Mlp, self).__init__(num_hidden_layers, num_neurons_hidden,
                                  input_size, output_size,
                                  learning_rate, batch_size, epochs, MlpLayer,
                                  h_activation, o_activation)

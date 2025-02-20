from NeuralLeet.nn.module import NeuralNetwork
import numpy as np


class mlp(NeuralNetwork):
    def __init__(
            self, number_of_layers: int, input_size: int,
            output_size: int, learning_rate: float = 0.01,
            batch_size: int = 128):

        # * Initialize the parent class NeuralNetwork
        super(mlp, self).__init__(number_of_layers, input_size, output_size,
                                  learning_rate, batch_size)

    # * Override the forward method
    def forward(self, x: np.ndarray):
        pass

    # * Override the backward method
    def backward(self, grad_output: np.ndarray):
        pass

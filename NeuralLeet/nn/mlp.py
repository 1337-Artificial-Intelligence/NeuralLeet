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
        - activation_function: callable
            The activation function to use for the layer
        - activation_function_derivative: callable
            The derivative of the activation function
        - learning_rate: float
            The learning rate of the neural network.
        - is_output_layer: bool
            A flag to indicate if the layer is the output
    '''

    def __init__(self, num_neurons: int, num_inputs: int,
                 activation_function: callable,
                 activation_function_derivative: callable,
                 learning_rate: float,
                 is_output_layer=False):
        '''
            Constructor for the MlpLayer class.
            Parameters:
                - num_neurons: int
                    The number of neurons in the layer.
                - num_inputs: int
                    The number of inputs to the each
                        neuron in the current layer.
                - activation_function: callable
                    The activation function to use for the layer
            Returns:
                None
        '''
        super().__init__(num_neurons, num_inputs, activation_function,
                         activation_function_derivative,
                         learning_rate, is_output_layer)

    def forward(self, x_batch: np.ndarray) -> np.ndarray:
        """
            Forward pass for the layer
            Parameters:
                - x_batch: np.ndarray
                    The input batch to the layer
            Returns:
                - np.ndarray
                    The output of the layer
        """
        output = np.dot(self.weights, x_batch) + self.bias
        activations = self.activation_function(output)
        self.last_input = x_batch
        if self.is_output_layer:
            self.last_output = activations
        else:
            self.last_output = output
        return activations

    def backward(self, y_true: np.ndarray,
                 weights_of_next_layer: np.ndarray,
                 last_error_derivative: np.ndarray = None) -> np.ndarray:
        """
        Backward pass with proper handling of activation derivatives

        Parameters:
            - y_true: True labels (only for output layer)
            - weights_of_next_layer: Weights of the next layer
            - last_error_derivative: Gradient from
                the next layer or output error
        """
        # ? d_z is the gradient of the loss with
        # ? respect to the output of the layer
        if self.is_output_layer:
            d_z = self.last_output - y_true
        else:
            d_z = np.dot(weights_of_next_layer.T, last_error_derivative) * \
                self.activation_function_derivative(self.last_output)

        # ? d_w is the gradient of the loss with
        # ? respect to the weights of the layer
        d_w = np.dot(d_z, self.last_input.T) / self.last_input.shape[1]

        # ? d_b is the gradient of the loss with
        # ?   respect to the bias of the layer
        d_b = np.sum(d_z, axis=1, keepdims=True) / self.last_input.shape[1]
        self.weights -= self.learning_rate * d_w
        self.bias -= self.learning_rate * d_b
        return d_z


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
        - h_activation_derivative: callable
            The derivative of the activation function for the hidden layers.
        - o_activation_derivative: callable
            The derivative of the activation function for the output layer.
        - loss: callable
            The loss function to use for training the neural network.
    '''

    def __init__(
            self, num_hidden_layers: int,
            num_neurons_hidden: int, input_size: int,
            h_activation: str, o_activation: str,
            output_size: int, learning_rate: float,
            batch_size: int, epochs: int,
            loss: callable
    ) -> None:

        # * Initialize the parent class NeuralNetwork
        super(Mlp, self).__init__(num_hidden_layers, num_neurons_hidden,
                                  input_size, output_size,
                                  learning_rate, batch_size, epochs, MlpLayer,
                                  h_activation,
                                  o_activation, loss)

    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        for e in range(self.epochs):
            for i in range(0, x.shape[1], self.batch_size):
                x_copy = x[:, i:i + self.batch_size]
                y_copy = y[:, i:i + self.batch_size]
                # * forward pass
                for layer in self.layers:
                    x_copy = layer.forward(x_copy)

                # * calculate loss
                loss = self.loss(x_copy, y_copy)

                # * calculate gradients
                last_layer_weights = None
                last_error_derivative = None
                for layer in reversed(self.layers):
                    last_error_derivative = layer.backward(
                        y_copy, last_layer_weights, last_error_derivative)
                    last_layer_weights = layer.weights
                if (e + 1) % 10 == 0:
                    print(f"Epoch {e+1}/{self.epochs}, Loss: {loss:.4f}")

    def predict(self, x: np.ndarray) -> np.ndarray:

        for layer in self.layers:
            x = layer.forward(x)
        return x

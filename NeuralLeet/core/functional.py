import numpy as np


class LossFunction:
    '''
        LossFunction is a class that represents a loss function.
        parameters:
            - name : str
                The name of the loss function.
            - function : callable
                The loss function.
            - derivative : callable
                The derivative of the loss function.
    '''

    def __init__(self, name, function, derivative):
        self.name = name
        self.function = function
        self.derivative = derivative


class ActivationFunction:
    '''
        ActivationFunction is a class that represents an activation function.
        parameters:
            - name : str
                The name of the activation function.
            - function : callable
                The activation function.
            - derivative : callable
                The derivative of the activation function.
    '''

    def __init__(self, name, function, derivative):
        self.name = name
        self.function = function
        self.derivative = derivative


def cross_entropy_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
    '''
        cross_entropy_loss is a function that calculates the cross entropy loss
            between the predictions and the targets.
        parameters:
            - predictions : np.ndarray
                The predictions of the neural network.
            - targets : np.ndarray
                The target values.
        returns:
            - float
                The cross entropy loss between the predictions and the targets.
    '''
    epsilon = 1e-12  # Small constant to avoid log(0)
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    logprobs = np.multiply(np.log(predictions), targets) + \
        ((1 - targets) * np.log(1 - predictions))
    cost = -np.sum(logprobs) / targets.shape[1]

    # makes sure cost is the dimension we expect.
    cost = float(np.squeeze(cost))

    return cost


def cross_entropy_derivative(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    '''
        cross_entropy_derivative is a function that calculates the derivative of
            the cross entropy loss function.
        parameters:
            - predictions : np.ndarray
                The predictions of the neural network.
            - targets : np.ndarray
                The target values.
        returns:
            - np.ndarray
                The derivative of the cross entropy loss function.
    '''
    epsilon = 1e-12  # Small constant to avoid division by zero
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    return -(targets / predictions) + ((1 - targets) / (1 - predictions))


def sigmoid(x: np.ndarray) -> np.ndarray:
    '''
        sigmoid is a function that calculates the sigmoid of the input.
        parameters:
            - x : np.ndarray
                The input to the sigmoid function.
        returns:
            - np.ndarray
                The output of the sigmoid function.
    '''
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    '''
        sigmoid_derivative is a function that calculates the derivative of
            the sigmoid function.
        parameters:
            - x : np.ndarray
                The input to the sigmoid function.
        returns:
            - np.ndarray
                The derivative of the sigmoid function.
    '''
    sig = sigmoid(x)
    return sig * (1 - sig)


def relu(x: np.ndarray) -> np.ndarray:
    '''
        relu is a function that returns the maximum of 0 and x,
        used as an activation function in neural networks.
        parameters:
            - x : np.ndarray
                The input to the activation function.
        returns:
            - np.ndarray
                The output of the activation function.
    '''
    """ReLU activation function"""
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    '''
        relu_derivative is a function that calculates the derivative of
            the relu function.
        parameters:
            - x : np.ndarray
                The input to the relu function.
        returns:
            - np.ndarray
                The derivative of the relu function.
    '''
    return np.where(x > 0, 1.0, 0.0)


def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    '''
        mse_loss is a function that calculates the mean squared error loss
            between the predictions and the targets.
        parameters:
            - y_pred : np.ndarray
                The predictions of the neural network.
            - y_true : np.ndarray
                The target values.
        returns:
            - float
                The mean squared error loss between the predictions and the targets.
    '''
    return np.mean(np.sum((y_pred - y_true) ** 2, axis=1))


def mse_derivative(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    '''
        mse_derivative is a function that calculates the derivative of
            the mean squared error loss function.
        parameters:
            - y_pred : np.ndarray
                The predictions of the neural network.
            - y_true : np.ndarray
                The target values.
        returns:
            - np.ndarray
                The derivative of the mean squared error loss function.
    '''
    return 2 * (y_pred - y_true)


# loss_functions = {
#     'cross_entropy': (cross_entropy_loss, cross_entropy_derivative),
#     'mse': (mse_loss, mse_derivative),
# }
# activation_functions = {
#     'sigmoid': (sigmoid, sigmoid_derivative),
#     'relu': (relu, relu_derivative),
# }

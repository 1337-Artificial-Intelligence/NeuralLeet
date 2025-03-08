import numpy as np


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
    logprobs = np.multiply(np.log(predictions), targets) + \
        ((1 - targets) * np.log(1 - predictions))
    cost = -np.sum(logprobs) / targets.shape[1]

    # makes sure cost is the dimension we expect.
    cost = float(np.squeeze(cost))

    return cost


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


def relu_derivative(x):
    """Derivative of ReLU function with respect to its input"""
    return np.where(x > 0, 1.0, 0.0)


def mse_loss(y_pred, y_true):
    """Calculate the mean squared error loss"""
    return np.mean(np.sum((y_pred - y_true) ** 2, axis=1))


activation_functions = {
    'sigmoid': (sigmoid, sigmoid_derivative),
    'relu': (relu, relu_derivative),
}

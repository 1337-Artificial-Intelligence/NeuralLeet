import numpy as np
import math
def relu(x_batch: np.ndarray) -> np.ndarray:
    '''
        relu is a function that returns the maximum of 0 and x, 
        used as an activation function in neural networks.
        parameters:
            - x_batch : np.ndarray
                The input to the activation function.
        returns:
            - np.ndarray
                The output of the activation function.
    '''
    return np.maximum(0, x_batch)



def softmax(predictions: np.ndarray) -> np.ndarray:
    '''
        softmax is a function that calculates the softmax of the predictions.
        parameters:
            - predictions : np.ndarray
                The predictions of the neural network.
        returns:
            - np.ndarray
                The softmax of the predictions.
    '''
    max_x = max(predictions)
    temp = [math.exp(x - max_x) for x in predictions]
    sum_temp = sum(temp)
    return [x / sum_temp for x in temp]


def log_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
    '''
        log_loss is a function that calculates the log loss between the predictions and the targets.
        parameters:
            - predictions : np.ndarray
                The predictions of the neural network.
            - targets : np.ndarray
                The target values.
        returns:
            - float
                The log loss between the predictions and the targets.
    '''

    losses = [-t * math.log(p) - (1 - t) * math.log(1 - p) for p, t in zip(predictions, targets)]
    return sum(losses)

import numpy as np
from typing import Callable
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
lambda_relu : Callable[[np.ndarray], np.ndarray] = lambda x_batch: np.maximum(0, x_batch)
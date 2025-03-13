
from NeuralLeet.nn.mlp import Mlp
import numpy as np
from NeuralLeet.core.functional import LossFunction, ActivationFunction, sigmoid, sigmoid_derivative, cross_entropy_loss, cross_entropy_derivative



inputs = [(0.0000, 0.3929), (0.5484, 0.7500), (0.0645, 0.5714), (0.5806, 0.5714),
          (0.2258, 0.8929), (0.4839, 0.2500), (0.3226, 0.2143), (0.7742, 0.8214),
          (0.4516, 0.5000), (0.4194, 0.0357), (0.4839, 0.2500), (0.3226, 0.7143),
          (0.5806, 0.5000), (0.5484, 0.1071), (0.6129, 0.6429), (0.6774, 0.1786),
          (0.2258, 0.8214), (0.7419, 0.1429), (0.6452, 1.0000), (0.8387, 0.2500),
          (0.9677, 0.3214), (0.3226, 0.4643), (0.3871, 0.5357), (0.3548, 0.1429),
          (0.3548, 0.6429), (0.1935, 0.4643), (0.4516, 0.3929), (0.4839, 0.6071),
          (0.6129, 0.6786), (0.2258, 0.6071), (0.5161, 0.3214), (0.5484, 0.6786),
          (0.3871, 0.8571), (0.6452, 0.6071), (0.1935, 0.3929), (0.6452, 0.3929),
          (0.6774, 0.4643), (0.3226, 0.2857), (0.7419, 0.7143), (0.7419, 0.3214),
          (1.0000, 0.3929), (0.8065, 0.3929), (0.1935, 0.5000), (0.1613, 0.8214),
          (0.2903, 0.9286), (0.3548, 0.0000), (0.2903, 0.6786), (0.5484, 0.9643),
          (0.4194, 0.1786), (0.2581, 0.2500), (0.3226, 0.7143), (0.5161, 0.3929),
          
          (0.2903, 0.6429), (0.5484, 0.9286), (0.2581, 0.3214), (0.0968, 0.5000),
          (0.6129, 0.7857), (0.0968, 0.3214), (0.6452, 0.9286), (0.8065, 0.7500)]

purple = (1, 0, 0)
orange = (0, 1, 0)
green = (0, 0, 1)

targets = [purple, orange, purple, orange, green, purple, purple, green, orange,
           purple, purple, green, orange, purple, orange, purple, green, purple, green,
           purple, purple, orange, orange, purple, orange, purple, orange, orange, orange,
           green, orange, orange, green, orange, purple, orange, orange, purple, orange,
           orange, purple, orange, green, green, green, purple, green, green, purple, purple,
           green, orange, green, green, purple, purple, green, purple, green, green]


sigmoid_activation = ActivationFunction("sigmoid", sigmoid, sigmoid_derivative)
cross_entropy = LossFunction("cross_entropy", cross_entropy_loss, cross_entropy_derivative)

_mlp = Mlp(num_hidden_layers=1, num_neurons_hidden=13, input_size=2,
           output_size=3, batch_size=128, epochs=10000, h_activation=sigmoid_activation,
           o_activation=sigmoid_activation,
           learning_rate=0.3, loss=cross_entropy
           )

input_t = np.array(inputs).T
target_t = np.array(targets).T


_mlp.train(input_t, target_t)

predictions = _mlp.predict(input_t)

predicted_classes = np.argmax(predictions, axis=0)
expected_classes = np.argmax(target_t, axis=0)

print("Accuracy:", np.mean(predicted_classes == expected_classes) * 100, "%")

# to run this code, run the following command in the terminal in the root directory of the project:
# python -m examples.main
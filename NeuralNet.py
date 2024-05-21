import nnfs
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

# Initialize nnfs to set random seeds and other configurations for reproducibility
nnfs.init()

# Define a class for a dense (fully connected) layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights with small random values and biases with zeros
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.output = None

    def forward(self, inputs):
        # Calculate the output of the dense layer using a dot product of inputs and weights, plus biases
        self.output = np.dot(inputs, self.weights) + self.biases

# Define a class for the ReLU activation function
class Activation_ReLU:
    def forward(self, inputs):
        # Apply the ReLU activation function (output is max(0, input))
        self.output = np.maximum(0, inputs)

# Define a class for the Softmax activation function
class Activation_Softmax:
    def forward(self, inputs):
        # Subtract the maximum value from each input for numerical stability (preventing overflow)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize the exponential values to get probabilities (sum of probabilities for each sample equals 1)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

# Generate a spiral dataset with 100 samples per class and 3 classes
X, y = spiral_data(100, 3)

# Create the first dense layer with 2 inputs (features) and 3 neurons
dense1 = Layer_Dense(2, 3)
# Create a ReLU activation object
activation1 = Activation_ReLU()

# Create the second dense layer with 3 inputs (from previous layer) and 3 neurons (one for each class)
dense2 = Layer_Dense(3, 3)
# Create a Softmax activation object
activation2 = Activation_Softmax()

# Perform the forward pass through the first dense layer
dense1.forward(X)
# Perform the forward pass through the ReLU activation function
activation1.forward(dense1.output)

# Perform the forward pass through the second dense layer
dense2.forward(activation1.output)
# Perform the forward pass through the Softmax activation function to get class probabilities
activation2.forward(dense2.output)

# Print the first 5 outputs (probabilities) from the final layer
print(activation2.output[:5])

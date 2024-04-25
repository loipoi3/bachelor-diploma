import numpy as np
from sklearn.metrics import log_loss
import time
from scipy.special import softmax


class MLP:
    """
    A simple implementation of a Multi-Layer Perceptron (MLP) neural network using manual mutations for training.

    Attributes:
        hidden_layer_sizes (list): List of integers specifying the number of neurons in each layer, including input and
        output layers.
        max_iter (int): Maximum number of iterations for training the network.
        _weights (list): List containing the weight matrices for each layer of the network.
        _errors (list): List that stores the log loss after each iteration during training.
        _errors_test (list): List that stores the log loss on the test set after each iteration (if applicable).
        _times (list): List storing the duration of each training iteration.
    """

    def __init__(self, hidden_layer_sizes: tuple, max_iter: int):
        """
        Initialize the MLP with the specified layer sizes and number of iterations.

        Parameters:
            hidden_layer_sizes (list): The number of neurons in each layer of the network.
            max_iter (int): The maximum number of iterations to run for training the model.
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter

        self._weights = []
        # Initialize weights using Xavier initialization for better convergence
        for idx in range(0, len(self.hidden_layer_sizes) - 1):
            xavier_stddev = np.sqrt(6.0 / (self.hidden_layer_sizes[idx] + self.hidden_layer_sizes[idx + 1]))
            self._weights.append(
                np.random.uniform(
                    -xavier_stddev,
                    xavier_stddev,
                    (self.hidden_layer_sizes[idx], self.hidden_layer_sizes[idx + 1])
                )
            )

        self._errors = None
        self._errors_test = None

    def _eval_model(self, X):
        """
        Evaluate the model's output for given input X.

        Parameters:
            X (array): Input features matrix.

        Returns:
            output (array): Output after passing through the MLP.
        """
        output = X
        for layer in range(len(self._weights) - 1):
            output = np.dot(output, self._weights[layer])
            output = 1 / (1 + np.exp(-output))
        if self._weights[-1].shape[1] == 1:
            output = np.dot(output, self._weights[-1])
            output = 1 / (1 + np.exp(-output))
        else:
            output = np.dot(output, self._weights[-1])
            output = softmax(output, axis=1)
        return output

    def fit(self, X, y, check_test_statistic: bool = False, X_test=None, y_test=None):
        """
        Fit the MLP model to the training data.

        Parameters:
            X (array): Training features.
            y (array): Training labels.
            check_test_statistic (bool): Whether to compute test statistics.
            X_test (array): Test features.
            y_test (array): Test labels.
        """
        # Initialize training error tracking
        self._errors = []
        self._times = []
        initial_loss = log_loss(y, self._eval_model(X))
        self._errors.append(initial_loss)
        if check_test_statistic:
            self._errors_test = []
            initial_test_loss = log_loss(y_test, self._eval_model(X_test))
            self._errors_test.append(initial_test_loss)
        for _ in range(self.max_iter):
            start_time = time.time()
            # Randomly select weight to mutate
            layer_index = np.random.randint(0, len(self._weights), size=1)[0]
            feature_index = np.random.randint(0, len(self._weights[layer_index]), size=1)[0]
            neuron_index = np.random.randint(0, len(self._weights[layer_index][feature_index]), size=1)[0]
            value_of_mutation = np.random.normal(scale=0.1)
            self._weights[layer_index][feature_index, neuron_index] += value_of_mutation
            current_loss = log_loss(y, self._eval_model(X))
            # Check if mutation improved the loss
            if current_loss < initial_loss:
                initial_loss = current_loss
            else:
                self._weights[layer_index][feature_index, neuron_index] -= value_of_mutation
            self._errors.append(initial_loss)
            if check_test_statistic:
                current_test_loss = log_loss(y_test, self._eval_model(X_test))
                self._errors_test.append(current_test_loss)
            self._times.append(time.time() - start_time)

    def predict(self, X, threshold=None, classes: int = 1):
        """
        Predict labels for given input X based on a threshold.

        Parameters:
            X (array): Input features.
            threshold (float): Threshold for converting probabilities to class labels.
            classes (int): Number of classes.

        Returns:
            y_pred (array): Predicted class labels.
        """
        output = self._eval_model(X)
        if classes == 1:
            y_pred = (output >= threshold).astype(int)
        else:
            y_pred = np.argmax(output, axis=1)
        return y_pred

import numpy as np
from sklearn.metrics import log_loss
import time


class MLP:
    def __init__(self, hidden_layer_sizes, max_iter):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter

        self._weights = []
        for idx in range(0, len(self.hidden_layer_sizes) - 1):
            xavier_stddev = np.sqrt(6.0 / (self.hidden_layer_sizes[idx] + self.hidden_layer_sizes[idx + 1]))
            self._weights.append(
                np.random.uniform(
                    -xavier_stddev,
                    xavier_stddev,
                    (self.hidden_layer_sizes[idx], self.hidden_layer_sizes[idx + 1])
                )
            )
        xavier_stddev = np.sqrt(6.0 / (self.hidden_layer_sizes[-1] + 1))
        self._weights.append(np.random.uniform(-xavier_stddev, xavier_stddev, (self.hidden_layer_sizes[-1], 1)))

        self._errors = None
        self._errors_test = None

    def _eval_model(self, X):
        output = X
        for layer in range(len(self._weights)):
            output = np.dot(output, self._weights[layer])
            output = 1 / (1 + np.exp(-output))
        return output

    def fit(self, X, y, check_test_statistic=False, X_test=None, y_test=None):
        # Training loop
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
            layer_index = np.random.randint(0, len(self._weights), size=1)[0]
            feature_index = np.random.randint(0, len(self._weights[layer_index]), size=1)[0]
            neuron_index = np.random.randint(0, len(self._weights[layer_index][feature_index]), size=1)[0]
            value_of_mutation = np.random.normal(scale=0.1)
            self._weights[layer_index][feature_index, neuron_index] += value_of_mutation
            current_loss = log_loss(y, self._eval_model(X))
            if current_loss < initial_loss:
                initial_loss = current_loss
            else:
                self._weights[layer_index][feature_index, neuron_index] -= value_of_mutation
            self._errors.append(initial_loss)
            if check_test_statistic:
                current_test_loss = log_loss(y_test, self._eval_model(X_test))
                self._errors_test.append(current_test_loss)
            self._times.append(time.time() - start_time)

    def predict(self, X, threshold):
        output = self._eval_model(X)
        y_pred = (output >= threshold).astype(int)
        return y_pred

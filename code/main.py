import torch
import torch.nn as nn
from torch.nn.functional import sigmoid
from sklearn.metrics import log_loss
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


class MLP(nn.Module):
    def __init__(self, hidden_layer_sizes):
        super(MLP, self).__init__()
        self.hidden_layer_sizes = hidden_layer_sizes
        self._layers = nn.ModuleList()

        # Initialization of layers
        for idx in range(len(self.hidden_layer_sizes) - 1):
            layer = nn.Linear(self.hidden_layer_sizes[idx], self.hidden_layer_sizes[idx + 1], bias=False)
            torch.nn.init.xavier_uniform_(layer.weight)
            self._layers.append(layer)

    def evaluate_model(self, X):
        output = X
        for layer in self._layers:
            output = sigmoid(layer(output))
        return output

    def set_weights(self, new_weights_):
        for layer, weights in zip(self._layers, new_weights_):
            layer.weight = nn.Parameter(torch.Tensor(weights))

    def get_weights(self):
        return [layer.weight.detach().numpy() for layer in self._layers]


def single_point_mutation(weights, scale=0.1):
    layer_index = np.random.randint(0, len(weights))
    feature_index = np.random.randint(0, len(weights[layer_index]))
    neuron_index = np.random.randint(0, len(weights[layer_index][feature_index]))
    value_of_mutation = np.random.normal(scale=scale)
    weights[layer_index][feature_index, neuron_index] += value_of_mutation
    return weights


diabetes = load_breast_cancer()
X, y = diabetes.data, diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = MLP(hidden_layer_sizes=[X.shape[1], 50, 100, 150, 100, 50, 1])

train_loss, test_loss = [], []
max_iter = 3000
initial_train_loss = log_loss(
    y_train.reshape(-1, 1),
    model.evaluate_model(torch.Tensor(X_train_scaled)).detach().numpy()
)
initial_test_loss = log_loss(
    y_test.reshape(-1, 1),
    model.evaluate_model(torch.Tensor(X_test_scaled)).detach().numpy()
)
train_loss.append(initial_train_loss)
test_loss.append(initial_test_loss)
for _ in range(max_iter):
    current_weights = model.get_weights()
    new_weights = single_point_mutation(current_weights)
    model.set_weights(new_weights)

    current_train_loss = log_loss(
        y_train.reshape(-1, 1),
        model.evaluate_model(torch.Tensor(X_train_scaled)).detach().numpy()
    )
    current_test_loss = log_loss(
        y_test.reshape(-1, 1),
        model.evaluate_model(torch.Tensor(X_test_scaled)).detach().numpy()
    )

    if current_train_loss < initial_train_loss:
        initial_train_loss = current_train_loss
    else:
        model.set_weights(current_weights)

    train_loss.append(initial_train_loss)
    test_loss.append(current_test_loss)

iterations = list(range(1, len(train_loss) + 1))
plt.plot(iterations, train_loss, label='Train Loss')
plt.plot(iterations, test_loss, label='Test Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss Over Iterations')
plt.legend()
plt.show()

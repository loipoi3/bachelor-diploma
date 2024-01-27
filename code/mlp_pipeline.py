from sklearn.datasets import load_breast_cancer
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
from models.mlp import MLP
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import warnings

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
np.random.seed(42)


def main_mlp():
    diabetes = load_breast_cancer()
    X, y = diabetes.data, diabetes.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = MLP(hidden_layer_sizes=(X.shape[1], 5, 10, 15, 10, 5, 1), max_iter=7000)
    model.fit(X_train_scaled, y_train, check_test_statistic=True, X_test=X_test_scaled, y_test=y_test)

    iterations = list(range(1, len(model._errors) + 1))
    plt.plot(iterations, model._errors, label='Train Loss')
    plt.plot(iterations, model._errors_test, label='Test Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss Over Iterations')
    plt.legend()
    plt.show()

    threshold = 0
    accuracy_lst, precision_lst, recall_lst, f1_lst = [], [], [], []
    while threshold <= 100:
        y_pred = model.predict(X_test_scaled, threshold=threshold)
        accuracy_lst.append((accuracy_score(y_test, y_pred), threshold))
        precision_lst.append((precision_score(y_test, y_pred), threshold))
        recall_lst.append((recall_score(y_test, y_pred), threshold))
        f1_lst.append((f1_score(y_test, y_pred), threshold))
        threshold += 0.01
    best_f1_score, best_threshold = max(f1_lst, key=lambda x: x[0])
    best_f1_index = f1_lst.index((best_f1_score, best_threshold))
    best_accuracy = accuracy_lst[best_f1_index][0]
    best_precision = precision_lst[best_f1_index][0]
    best_recall = recall_lst[best_f1_index][0]
    print("MLP with GA:")
    print(f"Best threshold={best_threshold}\nBest accuracy={best_accuracy}\nBest precision={best_precision}\n"
          f"Best recall={best_recall}\nBest f1-score={best_f1_score}\n")

    mlp_classifier = MLPClassifier(hidden_layer_sizes=(5, 10, 15, 10, 5,), max_iter=7000, random_state=42)
    mlp_classifier.fit(X_train_scaled, y_train)
    threshold = 0
    accuracy_lst, precision_lst, recall_lst, f1_lst = [], [], [], []
    while threshold <= 100:
        y_prob = mlp_classifier.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_prob > threshold).astype(int)
        accuracy_lst.append((accuracy_score(y_test, y_pred), threshold))
        precision_lst.append((precision_score(y_test, y_pred), threshold))
        recall_lst.append((recall_score(y_test, y_pred), threshold))
        f1_lst.append((f1_score(y_test, y_pred), threshold))
        threshold += 0.01
    best_f1_score, best_threshold = max(f1_lst, key=lambda x: x[0])
    best_f1_index = f1_lst.index((best_f1_score, best_threshold))
    best_accuracy = accuracy_lst[best_f1_index][0]
    best_precision = precision_lst[best_f1_index][0]
    best_recall = recall_lst[best_f1_index][0]
    print("Classic MLP:")
    print(f"Best threshold={best_threshold}\nBest accuracy={best_accuracy}\nBest precision={best_precision}\n"
          f"Best recall={best_recall}\nBest f1-score={best_f1_score}\n")


main_mlp()

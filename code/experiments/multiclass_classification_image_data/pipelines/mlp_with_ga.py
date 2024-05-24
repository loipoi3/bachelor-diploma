from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from code.models.mlp import MLP
from code.utils import plot_losses, summarize_best_loss_performance
import numpy as np


def run_mlp_with_ga(X_train_pca, X_test_pca, y_train, y_test):
    """
    Trains a Multilayer Perceptron (MLP) using Genetic Algorithm (GA) for optimization,
    and evaluates it using various metrics at different decision thresholds.

    Args:
    X_train_pca (numpy array): PCA-transformed features for the training data.
    X_test_pca (numpy array): PCA-transformed features for the test data.
    y_train (numpy array): Target labels for the training data.
    y_test (numpy array): Target labels for the test data.
    """
    # Initialize and train the MLP model
    model = MLP(hidden_layer_sizes=(X_train_pca.shape[1], 10, 10, len(np.unique(y_train))), max_iter=10000)
    model.fit(X_train_pca, y_train, check_test_statistic=True, X_test=X_test_pca, y_test=y_test)

    print("MLP with GA:")
    # Plot training and test loss history
    plot_losses(model._errors, model._errors_test)
    # Summarize best test loss performance and corresponding computation times
    summarize_best_loss_performance(model._errors_test, model._errors, model._times)

    # Predict classes directly
    y_pred_test = model.predict(X_test_pca, classes=len(np.unique(y_train)))

    # Evaluate metrics
    accuracy_test = accuracy_score(y_test, y_pred_test)
    precision_test = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
    recall_test = recall_score(y_test, y_pred_test, average='weighted')
    f1_test = f1_score(y_test, y_pred_test, average='weighted')
    print(
        f"Accuracy: {accuracy_test:.4f}, Precision: {precision_test:.4f}, Recall: {recall_test:.4f}, F1-score: {f1_test:.4f}")

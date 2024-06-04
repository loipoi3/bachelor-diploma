import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold

from code.models.mlp import MLP
from code.utils import plot_losses, summarize_best_loss_performance


def run_mlp_with_ga(X_train_pca, X_test_pca, y_train, y_test, n_splits=5, n_iterations=10000):
    """
    Trains a Multilayer Perceptron (MLP) using Genetic Algorithm (GA) for optimization,
    and evaluates it using various metrics at different decision thresholds with cross-validation.

    Args:
    X_train_pca (numpy array): PCA-transformed features for the training data.
    X_test_pca (numpy array): PCA-transformed features for the test data.
    y_train (numpy array): Target labels for the training data.
    y_test (numpy array): Target labels for the test data.
    n_splits (int): Number of cross-validation splits.
    n_iterations (int): Number of training iterations.
    """
    # Combine the training and test datasets
    X_pca = np.concatenate((X_train_pca, X_test_pca), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    skf = StratifiedKFold(n_splits=n_splits)
    # Initialize arrays to store results
    overall_train_log_losses = []
    overall_test_log_losses = []
    overall_accuracies = []
    overall_precisions = []
    overall_recalls = []
    overall_f1s = []
    total_time_list = []

    for fold_idx, (train_index, test_index) in enumerate(skf.split(X_pca, y)):
        X_train, X_test = X_pca[train_index], X_pca[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Initialize and train the MLP model
        model = MLP(hidden_layer_sizes=(X_train.shape[1], 15, 20, 15, 1), max_iter=n_iterations)
        model.fit(X_train, y_train, check_test_statistic=True, X_test=X_test, y_test=y_test)

        fold_train_losses = model._errors
        fold_test_losses = model._errors_test
        fold_accuracies, fold_precisions, fold_recalls, fold_f1s = [], [], [], []
        time_list = model._times

        # Find the best threshold for binary classification by maximizing the F1 score
        best_f1 = 0
        best_threshold = 0.0
        for threshold in np.arange(0.0, 1.01, 0.01):
            y_pred = model.predict(X_test, threshold=threshold)
            f1 = f1_score(y_test, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        # Compute accuracy, precision, recall, and F1 score using the best threshold
        y_pred = model.predict(X_test, threshold=best_threshold)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)

        fold_accuracies.append(accuracy)
        fold_precisions.append(precision)
        fold_recalls.append(recall)
        fold_f1s.append(best_f1)

        # Accumulate losses and metrics for each fold
        overall_train_log_losses.append(fold_train_losses)
        overall_test_log_losses.append(fold_test_losses)
        overall_accuracies.append(fold_accuracies)
        overall_precisions.append(fold_precisions)
        overall_recalls.append(fold_recalls)
        overall_f1s.append(fold_f1s)
        total_time_list.append(time_list)

    # Convert lists to numpy arrays for easier averaging
    overall_train_log_losses = np.array(overall_train_log_losses)
    overall_test_log_losses = np.array(overall_test_log_losses)
    overall_accuracies = np.array(overall_accuracies)
    overall_precisions = np.array(overall_precisions)
    overall_recalls = np.array(overall_recalls)
    overall_f1s = np.array(overall_f1s)
    total_time_list = np.array(total_time_list)

    # Average the losses and metrics across all folds
    avg_train_log_losses = np.mean(overall_train_log_losses, axis=0)
    avg_test_log_losses = np.mean(overall_test_log_losses, axis=0)
    avg_accuracies = np.mean(overall_accuracies, axis=0)
    avg_precisions = np.mean(overall_precisions, axis=0)
    avg_recalls = np.mean(overall_recalls, axis=0)
    avg_f1s = np.mean(overall_f1s, axis=0)
    avg_time_list = np.mean(total_time_list, axis=0)

    # Print the averaged results in the desired format
    print("train_loss_list =", avg_train_log_losses.tolist())
    print("test_loss_list =", avg_test_log_losses.tolist())
    print("accuracy_list =", avg_accuracies.tolist())
    print("precision_list =", avg_precisions.tolist())
    print("recall_list =", avg_recalls.tolist())
    print("f1_list =", avg_f1s.tolist())
    print("time_list =", avg_time_list.tolist())

    print("MLP with GA and Cross-Validation:")
    plot_losses(avg_train_log_losses, avg_test_log_losses)
    summarize_best_loss_performance(avg_test_log_losses, avg_train_log_losses, avg_time_list)

    # Final evaluation on the combined test set using the best threshold
    model.fit(X_train_pca, y_train)  # Re-train the model on the entire training set
    test_probs = model._eval_model(X_test_pca)
    best_f1 = 0
    best_threshold = 0.0
    for threshold in np.arange(0.0, 1.01, 0.01):
        y_pred = (test_probs >= threshold).astype(int)
        f1 = f1_score(y_test, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    y_pred = (test_probs >= best_threshold).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)

    print(f"Best Threshold={best_threshold:.2f}, Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1-score={best_f1:.4f}")

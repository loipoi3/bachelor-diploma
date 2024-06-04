import time
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from code.utils import plot_losses, summarize_best_loss_performance


def run_classic_mlp(X, y, n_splits=5, n_iterations=300):
    """
    Fit an MLP model and analyze its performance using cross-validation.

    Args:
    X (numpy array): Data features.
    y (numpy array): Data labels.
    n_splits (int): Number of cross-validation splits.
    n_iterations (int): Number of training iterations.
    """
    skf = StratifiedKFold(n_splits=n_splits)
    overall_train_log_losses, overall_test_log_losses = np.zeros(n_iterations), np.zeros(n_iterations)
    total_time_list = np.zeros(n_iterations)

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Initialize MLPClassifier model for iterative learning
        mlp = MLPClassifier(hidden_layer_sizes=(10, 15, 10), activation="tanh", solver="sgd", alpha=0.0008624588573893637,
                            learning_rate_init=0.003641245927855662, learning_rate="adaptive", max_iter=1, batch_size=32,
                            tol=1.57016158038566e-05, shuffle=False, early_stopping=False, warm_start=True)

        fold_train_losses, fold_test_losses = [], []
        time_list = []

        for i in range(n_iterations):
            start_time = time.time()
            mlp.fit(X_train, y_train)
            iteration_time = time.time() - start_time
            time_list.append(iteration_time)

            # Predict probabilities for both training and test sets
            train_probs = mlp.predict_proba(X_train)
            test_probs = mlp.predict_proba(X_test)

            # Compute log loss for the training and test sets
            train_loss = log_loss(y_train, train_probs)
            test_loss = log_loss(y_test, test_probs)

            fold_train_losses.append(train_loss)
            fold_test_losses.append(test_loss)

        # Accumulate losses for each fold
        overall_train_log_losses += np.array(fold_train_losses)
        overall_test_log_losses += np.array(fold_test_losses)
        total_time_list += np.array(time_list)

    # Average the losses across all folds
    avg_train_log_losses = overall_train_log_losses / n_splits
    avg_test_log_losses = overall_test_log_losses / n_splits
    avg_time_list = total_time_list / n_splits
    print("train_loss_list = ", avg_train_log_losses.tolist())
    print("test_loss_list = ", avg_test_log_losses.tolist())
    print("time_list = ", avg_time_list.tolist())
    print("Classic MLP with Cross-Validation:")
    plot_losses(avg_train_log_losses, avg_test_log_losses)
    _, res_time = summarize_best_loss_performance(avg_test_log_losses, avg_train_log_losses, avg_time_list)

    # Find the best threshold for binary classification by maximizing the F1 score
    threshold = 0.0
    accuracy_lst, precision_lst, recall_lst, f1_lst = [], [], [], []
    while threshold <= 1.0:
        y_prob = mlp.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > threshold).astype(int)
        accuracy_lst.append((accuracy_score(y_test, y_pred), threshold))
        precision_lst.append((precision_score(y_test, y_pred, zero_division=0), threshold))
        recall_lst.append((recall_score(y_test, y_pred), threshold))
        f1_lst.append((f1_score(y_test, y_pred), threshold))
        threshold += 0.01

    max_f1_score = max(f1_lst, key=lambda x: x[0])[0]
    best_thresholds = [threshold for f1, threshold in f1_lst if f1 == max_f1_score]

    # Output the best thresholds and corresponding performance metrics
    for th in best_thresholds:
        index = [i for i, (f1, threshold) in enumerate(f1_lst) if threshold == th][0]
        print(f"Threshold={th:.2f}, Accuracy={accuracy_lst[index][0]:.4f}, "
              f"Precision={precision_lst[index][0]:.4f}, Recall={recall_lst[index][0]:.4f}, "
              f"F1-score={f1_lst[index][0]:.4f}")
    return res_time

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from code.models.mlp import MLP
from code.utils import plot_losses, summarize_best_loss_performance


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
    model = MLP(hidden_layer_sizes=(X_train_pca.shape[1], 1), max_iter=30000)
    model.fit(X_train_pca, y_train, check_test_statistic=True, X_test=X_test_pca, y_test=y_test)

    print("MLP with GA:")
    # Plot training and test loss history
    plot_losses(model._errors, model._errors_test)
    # Summarize best test loss performance and corresponding computation times
    summarize_best_loss_performance(model._errors_test, model._times)

    # Initialize lists to store performance metrics at different thresholds
    threshold = 0.0
    accuracy_lst, precision_lst, recall_lst, f1_lst = [], [], [], []
    # Iterate through thresholds to find the best for classification based on F1 score
    while threshold <= 1.0:
        y_pred = model.predict(X_test_pca, threshold=threshold)
        accuracy_lst.append((accuracy_score(y_test, y_pred), threshold))
        precision_lst.append((precision_score(y_test, y_pred, zero_division=0), threshold))
        recall_lst.append((recall_score(y_test, y_pred), threshold))
        f1_lst.append((f1_score(y_test, y_pred), threshold))
        threshold += 0.01

    # Identify the maximum F1 score and corresponding best thresholds
    max_f1_score = max(f1_lst, key=lambda x: x[0])[0]
    best_thresholds = [threshold for f1, threshold in f1_lst if f1 == max_f1_score]

    # Output the best thresholds and corresponding performance metrics
    for th in best_thresholds:
        index = [i for i, (f1, threshold) in enumerate(f1_lst) if threshold == th][0]
        print(f"Threshold={th:.2f}, Accuracy={accuracy_lst[index][0]:.4f}, "
              f"Precision={precision_lst[index][0]:.4f}, Recall={recall_lst[index][0]:.4f}, "
              f"F1-score={f1_lst[index][0]:.4f}")

import time
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier

from code.utils import plot_losses, summarize_best_loss_performance


def run_classic_mlp(X_train_pca, X_test_pca, y_train, y_test):
    """
    Fit a Logistic Regression model and analyze its performance using PCA-transformed features.

    Args:
    X_train_pca (numpy array): Training data features after PCA transformation.
    X_test_pca (numpy array): Test data features after PCA transformation.
    y_train (numpy array): Training data labels.
    y_test (numpy array): Test data labels.
    """
    # Initialize LogisticRegression model for iterative learning
    mlp = MLPClassifier(hidden_layer_sizes=(), max_iter=1, warm_start=True)
    train_log_losses, test_log_losses = [], []
    n_iterations = 20
    time_list = []

    # Perform training over a set number of iterations to gather loss data
    for i in range(n_iterations):
        start_time = time.time()
        mlp.fit(X_train_pca, y_train)
        time_list.append(time.time() - start_time)

        # Predict probabilities for both training and test sets
        train_probs = mlp.predict_proba(X_train_pca)
        test_probs = mlp.predict_proba(X_test_pca)

        # Compute log loss for the training and test sets
        train_loss = log_loss(y_train, train_probs)
        test_loss = log_loss(y_test, test_probs)
        train_log_losses.append(train_loss)
        test_log_losses.append(test_loss)

    print("Classic MLP:")
    plot_losses(train_log_losses, test_log_losses)
    summarize_best_loss_performance(test_log_losses, time_list)

    # Find the best threshold for binary classification by maximizing the F1 score
    threshold = 0.0
    accuracy_lst, precision_lst, recall_lst, f1_lst = [], [], [], []
    while threshold <= 1.0:
        y_prob = mlp.predict_proba(X_test_pca)[:, 1]
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

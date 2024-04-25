import time
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from code.utils import plot_losses, summarize_best_loss_performance


def run_classic_mlp(X_train, X_test, y_train, y_test):
    """
    Fit a Logistic Regression model and analyze its performance using features.

    Args:
    X_train (numpy array): Training data features.
    X_test (numpy array): Test data features.
    y_train (numpy array): Training data labels.
    y_test (numpy array): Test data labels.
    """
    # Initialize LogisticRegression model for iterative learning
    mlp = MLPClassifier(hidden_layer_sizes=(), max_iter=1, warm_start=True)
    train_log_losses, test_log_losses = [], []
    n_iterations = 100
    time_list = []

    # Perform training over a set number of iterations to gather loss data
    for i in range(n_iterations):
        start_time = time.time()
        mlp.fit(X_train, y_train)
        time_list.append(time.time() - start_time)

        # Predict probabilities for both training and test sets
        train_probs = mlp.predict_proba(X_train)
        test_probs = mlp.predict_proba(X_test)

        # Compute log loss for the training and test sets
        train_loss = log_loss(y_train, train_probs)
        test_loss = log_loss(y_test, test_probs)
        train_log_losses.append(train_loss)
        test_log_losses.append(test_loss)

    print("Classic MLP:")
    plot_losses(train_log_losses, test_log_losses)
    summarize_best_loss_performance(test_log_losses, time_list)

    # Predict classes directly
    y_pred_test = mlp.predict(X_test)

    # Evaluate metrics
    accuracy_test = accuracy_score(y_test, y_pred_test)
    precision_test = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
    recall_test = recall_score(y_test, y_pred_test, average='weighted')
    f1_test = f1_score(y_test, y_pred_test, average='weighted')
    print(
        f"Accuracy: {accuracy_test:.4f}, Precision: {precision_test:.4f}, Recall: {recall_test:.4f}, F1-score: {f1_test:.4f}")

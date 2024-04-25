from code.models.one_plus_lambda_ea_with_gp_encodings import GeneticAlgorithmModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from code.utils import plot_losses, summarize_best_loss_performance


def run_one_plus_lambda_ea_with_gp(X_train, X_test, y_train, y_test):
    """
    Train and evaluate a genetic algorithm model with guided propagation on PCA-transformed data.

    Args:
    X_train_pca (numpy array): PCA-transformed features for the training data.
    X_test_pca (numpy array): PCA-transformed features for the test data.
    y_train (numpy array): Target labels for the training data.
    y_test (numpy array): Target labels for the test data.
    """
    # Initialize and run the genetic algorithm model
    model = GeneticAlgorithmModel(X_train, y_train, X_test, y_test, 4)
    champion, train_losses, test_losses, time_list = model.run(lambd=2, max_generations=400)

    print("(1 + lambda) - EA with GP:")
    # Plot the evolution of training and testing losses over generations
    plot_losses(train_losses, test_losses)
    # Summarize the performance in terms of test loss and computation time
    summarize_best_loss_performance(test_losses, time_list)

    # Analyze model predictions at various thresholds to maximize F1 score
    threshold = 0.0
    accuracy_lst, precision_lst, recall_lst, f1_lst = [], [], [], []
    while threshold <= 1.0:
        # Predict using the champion model at the current threshold
        y_pred = model.make_predictions_with_threshold(champion, X_test, threshold=threshold)
        # Evaluate and store different performance metrics
        accuracy_lst.append((accuracy_score(y_test, y_pred), threshold))
        precision_lst.append((precision_score(y_test, y_pred, zero_division=0), threshold))
        recall_lst.append((recall_score(y_test, y_pred), threshold))
        f1_lst.append((f1_score(y_test, y_pred), threshold))
        threshold += 0.01

    # Identify the threshold(s) that yielded the highest F1 score
    max_f1_score = max(f1_lst, key=lambda x: x[0])[0]
    best_thresholds = [threshold for f1, threshold in f1_lst if f1 == max_f1_score]

    # Output the best performance metrics at optimal thresholds
    for th in best_thresholds:
        index = [i for i, (f1, threshold) in enumerate(f1_lst) if threshold == th][0]
        print(f"Threshold={th:.2f}, Accuracy={accuracy_lst[index][0]:.4f}, "
              f"Precision={precision_lst[index][0]:.4f}, Recall={recall_lst[index][0]:.4f}, "
              f"F1-score={f1_lst[index][0]:.4f}")

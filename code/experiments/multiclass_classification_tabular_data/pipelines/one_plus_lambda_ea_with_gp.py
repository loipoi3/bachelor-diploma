from code.models.one_plus_lambda_ea_with_gp_encodings import GeneticAlgorithmModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from code.utils import plot_losses, summarize_best_loss_performance
import numpy as np


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
    model = GeneticAlgorithmModel(X_train, y_train, X_test, y_test, 6, num_classes=len(np.unique(y_train)))
    champion, train_losses, test_losses, time_list = model.run(
        lambd=2, max_generations=0, save_checkpoint=False,
        save_checkpoint_path="experiments/multiclass_classification_tabular_data/checkpoints/checkpoints_depth_6",
        start_checkpoint="experiments/multiclass_classification_tabular_data/checkpoints/checkpoints_depth_6/checkpoint_gen_14140.pkl"
    )

    print("(1 + lambda) - EA with GP:")
    # Plot the evolution of training and testing losses over generations
    plot_losses(train_losses, test_losses)
    # Summarize the performance in terms of test loss and computation time
    summarize_best_loss_performance(test_losses, time_list)

    # Predict classes directly
    y_pred_test = model.make_predictions_with_threshold(champion["ind"], X_test)

    # Evaluate metrics
    accuracy_test = accuracy_score(y_test, y_pred_test)
    precision_test = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
    recall_test = recall_score(y_test, y_pred_test, average='weighted')
    f1_test = f1_score(y_test, y_pred_test, average='weighted')
    print(
        f"Accuracy: {accuracy_test:.4f}, Precision: {precision_test:.4f}, Recall: {recall_test:.4f}, F1-score: {f1_test:.4f}")

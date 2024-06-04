import numpy as np
from sklearn.model_selection import StratifiedKFold

from code.models.one_plus_lambda_ea_with_gp_encodings import GeneticAlgorithmModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from code.utils import plot_losses, summarize_best_loss_performance


def run_one_plus_lambda_ea_with_gp(X_train_pca, X_test_pca, y_train, y_test, n_splits=5, n_iterations=3000):
    """
    Train and evaluate a genetic algorithm model with guided propagation on PCA-transformed data.

    Args:
    X_train_pca (numpy array): PCA-transformed features for the training data.
    X_test_pca (numpy array): PCA-transformed features for the test data.
    y_train (numpy array): Target labels for the training data.
    y_test (numpy array): Target labels for the test data.
    n_splits (int): Number of cross-validation splits.
    n_iterations (int): Number of iterations for the genetic algorithm.
    """
    # Combine the training and test datasets
    X_pca = np.concatenate((X_train_pca, X_test_pca), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    skf = StratifiedKFold(n_splits=n_splits)
    overall_train_log_losses = []
    overall_test_log_losses = []
    overall_accuracies = []
    overall_precisions = []
    overall_recalls = []
    overall_f1s = []
    total_time_list = []

    for fold_idx, (train_index, test_index) in enumerate(skf.split(X_pca, y)):
        print("fold_idx:", fold_idx)
        X_train, X_test = X_pca[train_index], X_pca[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Initialize and run the genetic algorithm model
        primitive_set = ["sub", "mul", "min", "max", "hypot", "_safe_atan2", "_float_lt", "_float_gt", "_float_ge", "_float_le"]
        terminal_set = ["Constant_0", "E"]
        model = GeneticAlgorithmModel(X_train, y_train, X_test, y_test, 6, primitive_set, terminal_set)
        champion, train_losses, test_losses, time_list, fold_f1s, fold_accuracies, fold_precisions, fold_recalls = model.run(lambd=4, max_generations=n_iterations, save_checkpoint_path="")

        fold_train_losses = train_losses
        fold_test_losses = test_losses

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

    # Average the losses and metrics across all folds for each iteration
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

    with open('output.txt', 'w') as file:
        # Write the lists to the file
        file.write("train_loss_list =" + str(avg_train_log_losses.tolist()) + "\n")
        file.write("test_loss_list =" + str(avg_test_log_losses.tolist()) + "\n")
        file.write("accuracy_list =" + str(avg_accuracies.tolist()) + "\n")
        file.write("precision_list =" + str(avg_precisions.tolist()) + "\n")
        file.write("recall_list =" + str(avg_recalls.tolist()) + "\n")
        file.write("f1_list =" + str(avg_f1s.tolist()) + "\n")
        file.write("time_list =" + str(avg_time_list.tolist()) + "\n")

    print("(1 + lambda) - EA with GP and Cross-Validation:")
    plot_losses(avg_train_log_losses, avg_test_log_losses)
    summarize_best_loss_performance(avg_test_log_losses, avg_train_log_losses, avg_time_list)

    # Final evaluation on the combined test set using the best threshold
    model = GeneticAlgorithmModel(X_train_pca, y_train, X_test_pca, y_test, 6, primitive_set, terminal_set)
    champion, _, _, _ = model.run(lambd=4, max_generations=n_iterations)
    best_f1 = 0
    best_threshold = 0.0
    for threshold in np.arange(0.0, 1.01, 0.01):
        y_pred = model.make_predictions_with_threshold(champion, X_test_pca, threshold=threshold)
        f1 = f1_score(y_test, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    y_pred = model.make_predictions_with_threshold(champion, X_test_pca, threshold=best_threshold)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)

    print(f"Best Threshold={best_threshold:.2f}, Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1-score={best_f1:.4f}")

# import optuna
# from itertools import combinations
#
#
# def run_one_plus_lambda_ea_with_gp(X_train, X_test, y_train, y_test, mlp_time):
#     def objective(trial):
#         # Define the search space using the trial object
#         depth = trial.suggest_int('depth', 1, 8)
#         lambd = trial.suggest_int('lambd', 0, 4)
#         max_generations = trial.suggest_int('max_generations', 100, 1000)
#
#         # Possible configurations for primitives and terminals
#         primitives = ["add", "sub", "mul", "_safe_div", "min", "max", "hypot", "logaddexp", "_safe_atan2", "_float_lt",
#                       "_float_gt", "_float_ge", "_float_le", "_safe_fmod"]
#         terminals = ["Constant_0", "Constant_1", "Constant_minus_1", "Pi", "E"]
#
#         def encode_combinations(combinations):
#             """Encode each combination of items into a single string."""
#             return ["|".join(combo) for combo in combinations]
#
#         def decode_combination(encoded_combination):
#             """Decode the encoded string back into a list."""
#             return encoded_combination.split("|")
#
#         # Precompute and encode all combinations of different lengths
#         all_primitive_combinations = [list(combo) for i in range(1, len(primitives) + 1) for combo in
#                                       combinations(primitives, i)]
#         all_terminal_combinations = [list(combo) for i in range(1, len(terminals) + 1) for combo in
#                                      combinations(terminals, i)]
#
#         encoded_primitive_combinations = encode_combinations(all_primitive_combinations)
#         encoded_terminal_combinations = encode_combinations(all_terminal_combinations)
#
#         # Suggest a combination of encoded primitives and terminals
#         encoded_primitive_set = trial.suggest_categorical('primitive_set', encoded_primitive_combinations)
#         encoded_terminal_set = trial.suggest_categorical('terminal_set', encoded_terminal_combinations)
#
#         # Decode the combinations
#         primitive_set = decode_combination(encoded_primitive_set)
#         terminal_set = decode_combination(encoded_terminal_set)
#
#         # Initialize and run the genetic algorithm model
#         model = GeneticAlgorithmModel(X_train, y_train, X_test, y_test, depth, primitive_set, terminal_set)
#         champion, train_losses, test_losses, time_list = model.run(lambd=lambd, max_generations=max_generations,
#                                                                    save_checkpoint_path="./", mlp_time=mlp_time)
#
#         # Extract the best test loss
#         loss, _ = summarize_best_loss_performance(test_losses, time_list)
#
#         return loss
#
#     def run_optimization():
#         study = optuna.create_study(direction='minimize')
#         study.optimize(objective, n_trials=1000)
#
#         print('Best parameters:', study.best_params)
#         print('Best loss:', study.best_value)
#
#         return study
#
#     study = run_optimization()
#
#     optuna.visualization.plot_optimization_history(study)
#     optuna.visualization.plot_param_importances(study)

from code.models.one_plus_lambda_ea_with_gp_encodings import GeneticAlgorithmModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from code.utils import plot_losses, summarize_best_loss_performance


def run_one_plus_lambda_ea_with_gp(X_train_pca, X_test_pca, y_train, y_test, mlp_time):
    """
    Train and evaluate a genetic algorithm model with guided propagation on PCA-transformed data.

    Args:
    X_train_pca (numpy array): PCA-transformed features for the training data.
    X_test_pca (numpy array): PCA-transformed features for the test data.
    y_train (numpy array): Target labels for the training data.
    y_test (numpy array): Target labels for the test data.
    """
    # Initialize and run the genetic algorithm model
    primitive_set = ["sub", "mul", "min", "max", "hypot", "_safe_atan2", "_float_lt", "_float_gt", "_float_ge",
                     "_float_le"]
    terminal_set = ["Constant_0", "E"]
    model = GeneticAlgorithmModel(X_train_pca, y_train, X_test_pca, y_test, 6, primitive_set, terminal_set)
    champion, train_losses, test_losses, time_list = model.run(lambd=4, max_generations=0,
                                                               save_checkpoint_path="",
                                                               start_checkpoint="experiments/binary_classification_image_data/checkpoints/lambda_4_depth_6_pca/checkpoint_gen_48041.pkl",
                                                               save_checkpoint=False)

    print("(1 + lambda) - EA with GP:")
    # Plot the evolution of training and testing losses over generations
    plot_losses(train_losses, test_losses)
    # Summarize the performance in terms of test loss and computation time
    summarize_best_loss_performance(test_losses, train_losses, time_list)

    # Analyze model predictions at various thresholds to maximize F1 score
    threshold = 0.0
    accuracy_lst, precision_lst, recall_lst, f1_lst = [], [], [], []
    while threshold <= 1.0:
        # Predict using the champion model at the current threshold
        y_pred = model.make_predictions_with_threshold(champion, X_test_pca, threshold=threshold)
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

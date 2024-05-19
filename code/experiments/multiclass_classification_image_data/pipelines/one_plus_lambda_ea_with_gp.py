from code.models.one_plus_lambda_ea_with_gp_encodings import GeneticAlgorithmModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from code.utils import plot_losses, summarize_best_loss_performance
import numpy as np


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
    primitive_set = ["add", "sub", "mul", "_safe_div", "min", "max", "hypot", "logaddexp"]
    terminal_set = ["Constant_0", "Constant_1", "Constant_minus_1"]
    model = GeneticAlgorithmModel(X_train_pca, y_train, X_test_pca, y_test, 8, primitive_set, terminal_set,
                                  num_classes=len(np.unique(y_train)))
    champion, train_losses, test_losses, time_list = model.run(lambd=4, max_generations=0,
                                                               save_checkpoint_path="",
                                                               start_checkpoint="experiments/multiclass_classification_image_data/checkpoints/lambda_4_depth_8_pca/checkpoint_gen_4937.pkl",
                                                               save_checkpoint=False)

    print("(1 + lambda) - EA with GP:")
    # Plot the evolution of training and testing losses over generations
    plot_losses(train_losses, test_losses)
    # Summarize the performance in terms of test loss and computation time
    summarize_best_loss_performance(test_losses, time_list)

    # Predict classes directly
    y_pred_test = model.make_predictions_with_threshold(champion, X_test_pca)

    # Evaluate metrics
    accuracy_test = accuracy_score(y_test, y_pred_test)
    precision_test = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
    recall_test = recall_score(y_test, y_pred_test, average='weighted')
    f1_test = f1_score(y_test, y_pred_test, average='weighted')
    print(
        f"Accuracy: {accuracy_test:.4f}, Precision: {precision_test:.4f}, Recall: {recall_test:.4f}, F1-score: {f1_test:.4f}")

    # for tree in champion["ind"]:
    #     nodes, edges, labels = gp.graph(tree)
    #
    #     g = nx.Graph()
    #     g.add_nodes_from(nodes)
    #     g.add_edges_from(edges)
    #     pos = graphviz_layout(g, prog="dot")
    #
    #     nx.draw_networkx_nodes(g, pos)
    #     nx.draw_networkx_edges(g, pos)
    #     nx.draw_networkx_labels(g, pos, labels)
    #     plt.show()

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
#         model = GeneticAlgorithmModel(X_train, y_train, X_test, y_test, depth, primitive_set, terminal_set,
#                                       num_classes=len(np.unique(y_train)))
#         champion, train_losses, test_losses, time_list = model.run(lambd=lambd, max_generations=max_generations,
#                                                                    save_checkpoint_path="./")
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

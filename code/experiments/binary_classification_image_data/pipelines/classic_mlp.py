import time

import numpy as np
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from code.utils import plot_losses, summarize_best_loss_performance


def run_classic_mlp(X_train_pca, X_test_pca, y_train, y_test, n_splits=5, n_iterations=500):
    """
    Fit an MLP model and analyze its performance using PCA-transformed features with cross-validation.

    Args:
    X_train_pca (numpy array): Training data features after PCA transformation.
    X_test_pca (numpy array): Test data features after PCA transformation.
    y_train (numpy array): Training data labels.
    y_test (numpy array): Test data labels.
    n_splits (int): Number of cross-validation splits.
    n_iterations (int): Number of training iterations.
    """
    # Combine the training and test datasets
    X_pca = np.concatenate((X_train_pca, X_test_pca), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    skf = StratifiedKFold(n_splits=n_splits)
    overall_train_log_losses, overall_test_log_losses = np.zeros((n_splits, n_iterations)), np.zeros((n_splits, n_iterations))
    overall_accuracies, overall_precisions, overall_recalls, overall_f1s = np.zeros((n_splits, n_iterations)), np.zeros((n_splits, n_iterations)), np.zeros((n_splits, n_iterations)), np.zeros((n_splits, n_iterations))
    total_time_list = np.zeros(n_iterations)

    for fold_idx, (train_index, test_index) in enumerate(skf.split(X_pca, y)):
        X_train, X_test = X_pca[train_index], X_pca[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Initialize MLPClassifier model for iterative learning
        mlp = MLPClassifier(hidden_layer_sizes=(15, 20, 15), activation="tanh", solver="sgd", alpha=0.005383724166734261,
                            learning_rate_init=0.0015898533701208645, learning_rate="invscaling", batch_size=256,
                            max_iter=1, early_stopping=False, tol=0.00032994812784605145, shuffle=True, warm_start=True)

        fold_train_losses, fold_test_losses = [], []
        fold_accuracies, fold_precisions, fold_recalls, fold_f1s = [], [], [], []
        time_list = []

        for i in range(1, n_iterations + 1):
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

            # Find the best threshold for binary classification by maximizing the F1 score
            best_f1 = 0
            best_threshold = 0.0
            for threshold in np.arange(0.0, 1.01, 0.01):
                y_pred = (test_probs[:, 1] > threshold).astype(int)
                f1 = f1_score(y_test, y_pred)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold

            # Compute accuracy, precision, recall, and F1 score using the best threshold
            y_pred = (test_probs[:, 1] > best_threshold).astype(int)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred)

            fold_accuracies.append(accuracy)
            fold_precisions.append(precision)
            fold_recalls.append(recall)
            fold_f1s.append(best_f1)

        # Accumulate losses and metrics for each fold
        overall_train_log_losses[fold_idx] = np.array(fold_train_losses)
        overall_test_log_losses[fold_idx] = np.array(fold_test_losses)
        overall_accuracies[fold_idx] = np.array(fold_accuracies)
        overall_precisions[fold_idx] = np.array(fold_precisions)
        overall_recalls[fold_idx] = np.array(fold_recalls)
        overall_f1s[fold_idx] = np.array(fold_f1s)
        total_time_list += np.array(time_list)

    # Average the losses and metrics across all folds
    avg_train_log_losses = np.mean(overall_train_log_losses, axis=0)
    avg_test_log_losses = np.mean(overall_test_log_losses, axis=0)
    avg_accuracies = np.mean(overall_accuracies, axis=0)
    avg_precisions = np.mean(overall_precisions, axis=0)
    avg_recalls = np.mean(overall_recalls, axis=0)
    avg_f1s = np.mean(overall_f1s, axis=0)
    avg_time_list = total_time_list / n_splits

    # Print the averaged results in the desired format
    print("train_loss_list =", avg_train_log_losses.tolist())
    print("test_loss_list =", avg_test_log_losses.tolist())
    print("accuracy_list =", avg_accuracies.tolist())
    print("precision_list =", avg_precisions.tolist())
    print("recall_list =", avg_recalls.tolist())
    print("f1_list =", avg_f1s.tolist())
    print("time_list =", avg_time_list.tolist())

    print("Classic MLP with Cross-Validation:")
    plot_losses(avg_train_log_losses, avg_test_log_losses)
    summarize_best_loss_performance(avg_test_log_losses, avg_train_log_losses, avg_time_list)

    # Final evaluation on the combined test set using the best threshold
    test_probs = mlp.predict_proba(X_test_pca)
    best_f1 = 0
    best_threshold = 0.0
    for threshold in np.arange(0.0, 1.01, 0.01):
        y_pred = (test_probs[:, 1] > threshold).astype(int)
        f1 = f1_score(y_test, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    y_pred = (test_probs[:, 1] > best_threshold).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)

    print(f"Best Threshold={best_threshold:.2f}, Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1-score={best_f1:.4f}")


# import optuna
#
#
# def run_classic_mlp(X_train, X_test, y_train, y_test):
#     def objective(trial):
#         hidden_layer_sizes = trial.suggest_categorical('hidden_layer_sizes',
#                                                        ['()', '(10, 10)', '(10, 15, 10)',
#                                                         '(15, 20, 15)', '(10, 15, 20, 15, 10)'])
#         hidden_layer_sizes = eval(hidden_layer_sizes)
#
#         params = {
#             'hidden_layer_sizes': hidden_layer_sizes,
#             'activation': trial.suggest_categorical('activation', ['identity', 'logistic', 'tanh', 'relu']),
#             'solver': trial.suggest_categorical('solver', ['lbfgs', 'sgd', 'adam']),
#             'alpha': trial.suggest_float('alpha', 0.0001, 0.01, log=True),
#             'learning_rate_init': trial.suggest_float('learning_rate_init', 0.001, 0.01, log=True),
#             'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive']),
#             'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
#             'max_iter': trial.suggest_int('max_iter', 10, 1000),
#             'early_stopping': trial.suggest_categorical('early_stopping', [True, False]),
#             'tol': trial.suggest_float('tol', 1e-6, 1e-3, log=True),
#             'shuffle': trial.suggest_categorical('shuffle', [True, False])
#         }
#
#         mlp = MLPClassifier(**params)
#
#         # Fit the model
#         mlp.fit(X_train, y_train)
#
#         # Predict probabilities on the test set
#         probs = mlp.predict_proba(X_test)
#
#         # Use log loss as the objective to minimize
#         return log_loss(y_test, probs)
#
#     # Create a study object and specify the direction of the optimization
#     study = optuna.create_study(direction='minimize')
#     study.optimize(objective, n_trials=1000)
#
#     # Print the result
#     print('Best trial:')
#     trial = study.best_trial
#     print(f'  Value: {trial.value}')
#     print('  Params: ')
#     for key, value in trial.params.items():
#         print(f'    {key}: {value}')

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
    mlp = MLPClassifier(hidden_layer_sizes=(), activation="logistic", solver="sgd", alpha=0.00018245304327849503,
                        learning_rate_init=0.0015409633841540953, learning_rate="adaptive", batch_size=64,
                        early_stopping=True, tol=7.717803388276812e-05, shuffle=False, max_iter=1, warm_start=True)
    train_log_losses, test_log_losses = [], []
    n_iterations = 932
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
    _, res_time = summarize_best_loss_performance(test_log_losses, time_list)

    # Predict classes directly
    y_pred_test = mlp.predict(X_test)

    # Evaluate metrics
    accuracy_test = accuracy_score(y_test, y_pred_test)
    precision_test = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
    recall_test = recall_score(y_test, y_pred_test, average='weighted')
    f1_test = f1_score(y_test, y_pred_test, average='weighted')
    print(
        f"Accuracy: {accuracy_test:.4f}, Precision: {precision_test:.4f}, Recall: {recall_test:.4f}, F1-score: {f1_test:.4f}")

    return res_time

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

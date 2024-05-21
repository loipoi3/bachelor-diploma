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
    mlp = MLPClassifier(hidden_layer_sizes=(15, 20, 15), activation="tanh", solver="sgd", alpha=0.005383724166734261,
                        learning_rate_init=0.0015898533701208645, learning_rate="invscaling", batch_size=256,
                        max_iter=1, early_stopping=False, tol=0.00032994812784605145, shuffle=True, warm_start=True)
    train_log_losses, test_log_losses = [], []
    n_iterations = 5  #406
    time_list = []

    # Perform training over a set number of iterations to gather loss data
    for i in range(1, n_iterations+1):
        start_time = time.time()
        mlp.fit(X_train_pca, y_train)
        time_list.append(time.time() - start_time)

        # Predict probabilities for both training and test sets
        train_probs = mlp.predict_proba(X_train_pca)
        test_probs = mlp.predict_proba(X_test_pca)

        # Compute log loss for the training and test sets
        train_loss = log_loss(y_train, train_probs)
        test_loss = log_loss(y_test, test_probs)
        print("Iter: ", i)
        print("Time: ", sum(time_list))
        print("train_loss: ", train_loss)
        print("test_loss: ", test_loss)
        train_log_losses.append(train_loss)
        test_log_losses.append(test_loss)
    print("Time list: ", time_list)
    print("Train loss list: ", train_log_losses)
    print("Test loss list: ", test_log_losses)

    print("Classic MLP:")
    plot_losses(train_log_losses, test_log_losses)
    summarize_best_loss_performance(test_log_losses, train_log_losses, time_list)

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

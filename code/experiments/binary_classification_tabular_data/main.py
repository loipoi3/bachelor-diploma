import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
import warnings
from sklearn.exceptions import ConvergenceWarning
# from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
from pipelines.mlp_with_ga import run_mlp_with_ga
from pipelines.classic_mlp import run_classic_mlp
from pipelines.one_plus_lambda_ea_with_gp import run_one_plus_lambda_ea_with_gp
from sklearn.neighbors import LocalOutlierFactor
# import optuna

# Suppress convergence warnings which are common in iterative algorithms if they stop early
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def median_target(var):
    temp = df[df[var].notnull()]
    temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()
    return temp


# def objective(trial):
#     params = {
#         'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes',
#                                                         [(10, 10), (10, 15, 10), (15, 20, 15), (10, 15, 20, 15, 10)]),
#         'activation': trial.suggest_categorical('activation', ['identity', 'logistic', 'tanh', 'relu']),
#         'solver': trial.suggest_categorical('solver', ['lbfgs', 'sgd', 'adam']),
#         'alpha': trial.suggest_loguniform('alpha', 0.0001, 0.01),
#         'learning_rate_init': trial.suggest_loguniform('learning_rate_init', 0.001, 0.01),
#         'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive']),
#         'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
#         'max_iter': trial.suggest_int('max_iter', 100, 1000),
#         'early_stopping': trial.suggest_categorical('early_stopping', [True, False]),
#         'tol': trial.suggest_loguniform('tol', 1e-6, 1e-3),
#         'shuffle': trial.suggest_categorical('shuffle', [True, False])
#     }
#
#     mlp = MLPClassifier(**params)
#     score = cross_val_score(mlp, X_train_scaled, y_train, n_jobs=-1, cv=3).mean()
#     return score


if __name__ == "__main__":
    # Load dataset containing embeddings, for a task image classification
    df = pd.read_csv('datasets/pima_indians_diabetes_database/diabetes.csv')

    df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction',
        'Age']] = df[
        ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction',
         'Age']].replace(0, np.NaN)

    columns = df.columns
    columns = columns.drop("Outcome")
    for i in columns:
        median_target(i)
        df.loc[(df['Outcome'] == 0) & (df[i].isnull()), i] = median_target(i)[i][0]
        df.loc[(df['Outcome'] == 1) & (df[i].isnull()), i] = median_target(i)[i][1]

    Q1 = df.Insulin.quantile(0.25)
    Q3 = df.Insulin.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df.loc[df['Insulin'] > upper, "Insulin"] = upper

    lof = LocalOutlierFactor(n_neighbors=10)
    lof.fit_predict(df)

    df_scores = lof.negative_outlier_factor_

    thresold = np.sort(df_scores)[7]

    outlier = df_scores > thresold

    df = df[outlier]

    # Split features and labels
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Split data into training and testing sets with a test size of 20%
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)
    # study = optuna.create_study(direction='maximize')
    # study.optimize(objective, n_trials=1000)
    #
    # print("Best parameters:", study.best_trial.params)
    # print("Best score:", study.best_trial.value)
    # Run the different ML model pipelines with the processed data
    res_time = run_classic_mlp(X_scaled, y)
    # print()
    # run_mlp_with_ga(X_train_scaled, X_test_scaled, y_train, y_test)
    # print()
    # run_one_plus_lambda_ea_with_gp(X_train_scaled, X_test_scaled, y_train, y_test, res_time)

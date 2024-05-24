import pandas as pd
import numpy as np
import warnings
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pipelines.mlp_with_ga import run_mlp_with_ga
from pipelines.classic_mlp import run_classic_mlp
from pipelines.one_plus_lambda_ea_with_gp import run_one_plus_lambda_ea_with_gp

# Suppress convergence warnings which are common in iterative algorithms if they stop early
warnings.filterwarnings("ignore", category=ConvergenceWarning)

if __name__ == "__main__":
    # Load dataset containing embeddings, for a task image classification
    df_train = pd.read_csv('datasets/human_activity_recognition_with_smartphones/train.csv')
    df_test = pd.read_csv('datasets/human_activity_recognition_with_smartphones/test.csv')

    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()

    # Fit the label encoder on the training 'Activity' data and transform it
    df_train['Activity'] = label_encoder.fit_transform(df_train['Activity'])

    # Transform the testing 'Activity' data using the fitted encoder
    df_test['Activity'] = label_encoder.transform(df_test['Activity'])

    # Split features and labels for training data
    X_train = df_train.iloc[:, :-1].values
    y_train = df_train.iloc[:, -1].values

    # Split features and labels for testing data
    X_test = df_test.iloc[:, :-1].values
    y_test = df_test.iloc[:, -1].values

    # Shuffle training data
    shuffle_index_train = np.random.permutation(len(X_train))
    X_train = X_train[shuffle_index_train]
    y_train = y_train[shuffle_index_train]

    # Shuffle testing data
    shuffle_index_test = np.random.permutation(len(X_test))
    X_test = X_test[shuffle_index_test]
    y_test = y_test[shuffle_index_test]

    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # pca = PCA(n_components=0.99)
    # X_train_pca = pca.fit_transform(X_train_scaled)
    # X_test_pca = pca.transform(X_test_scaled)

    # Run the different ML model pipelines with the processed data
    res_time = run_classic_mlp(X_train_scaled, X_test_scaled, y_train, y_test)
    # print()
    # run_mlp_with_ga(X_train_scaled, X_test_scaled, y_train, y_test)
    # print()
    # run_one_plus_lambda_ea_with_gp(X_train_scaled, X_test_scaled, y_train, y_test, 1)

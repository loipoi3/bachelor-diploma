import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# from pipelines.mlp_with_ga import run_mlp_with_ga
from pipelines.classic_mlp import run_classic_mlp
# from pipelines.one_plus_lambda_ea_with_gp import run_one_plus_lambda_ea_with_gp
import numpy as np

np.random.seed(42)

# Suppress convergence warnings which are common in iterative algorithms if they stop early
warnings.filterwarnings("ignore", category=ConvergenceWarning)

if __name__ == "__main__":
    # Load dataset containing embeddings, for a task image classification
    df_train = pd.read_csv('datasets/intel_image_classification/seg_train/train_embeddings.csv')
    # Split features and labels
    X_train = df_train.iloc[:, :-1].values
    y_train = df_train.iloc[:, -1].values
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]

    # Load dataset containing embeddings, for a task image classification
    df_test = pd.read_csv('datasets/intel_image_classification/seg_test/test_embeddings.csv')
    # Split features and labels
    X_test = df_test.iloc[:, :-1].values
    y_test = df_test.iloc[:, -1].values
    indices = np.arange(X_test.shape[0])
    np.random.shuffle(indices)
    X_test = X_test[indices]
    y_test = y_test[indices]

    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reduce dimensionality using PCA to retain 95% of variance
    # pca = PCA(n_components=0.95)
    # X_train_pca = pca.fit_transform(X_train_scaled)
    # X_test_pca = pca.transform(X_test_scaled)

    # Run the different ML model pipelines with the processed data
    run_classic_mlp(X_train_scaled, X_test_scaled, y_train, y_test)
    # print()
    # run_mlp_with_ga(X_train_pca, X_test_pca, y_train, y_test)
    # print()
    # run_one_plus_lambda_ea_with_gp(X_train_pca, X_test_pca, y_train, y_test)

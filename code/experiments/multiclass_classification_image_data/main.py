import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pipelines.mlp_with_ga import run_mlp_with_ga
from pipelines.classic_mlp import run_classic_mlp
from pipelines.one_plus_lambda_ea_with_gp import run_one_plus_lambda_ea_with_gp

# Suppress convergence warnings which are common in iterative algorithms if they stop early
warnings.filterwarnings("ignore", category=ConvergenceWarning)

if __name__ == "__main__":
    # Load dataset containing embeddings, for a task image classification
    df_train = pd.read_csv('datasets/chest_xray/train_embeddings_multiclass.csv')
    df_test = pd.read_csv('datasets/chest_xray/test_embeddings_multiclass.csv')
    # Split features and labels
    X_train = df_train.iloc[:, :-1].values
    y_train = df_train.iloc[:, -1].values
    X_test = df_test.iloc[:, :-1].values
    y_test = df_test.iloc[:, -1].values

    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # # Reduce dimensionality using PCA to retain 95% of variance
    pca = PCA(n_components=0.99)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Run the different ML model pipelines with the processed data
    # run_classic_mlp(X_train_pca, X_test_pca, y_train, y_test)
    # print()
    # run_mlp_with_ga(X_train_pca, X_test_pca, y_train, y_test)
    # print()
    run_one_plus_lambda_ea_with_gp(X_train_pca, X_test_pca, y_train, y_test, 1)

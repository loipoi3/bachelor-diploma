import pandas as pd
from sklearn.model_selection import train_test_split
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
    df = pd.read_csv('datasets/cats_vs_dogs/embeddings.csv')
    # Split features and labels
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Split data into training and testing sets with a test size of 20%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reduce dimensionality using PCA to retain 95% of variance
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Run the different ML model pipelines with the processed data
    # run_classic_mlp(X_train_pca, X_test_pca, y_train, y_test)
    # print()
    # run_mlp_with_ga(X_train_pca, X_test_pca, y_train, y_test)
    # print()
    run_one_plus_lambda_ea_with_gp(X_train_pca, X_test_pca, y_train, y_test)

# import torch
# import torchvision.models as models
# import torchvision.transforms as transforms
# from PIL import Image
# from torch.utils.data import Dataset, DataLoader
# from pathlib import Path
# import numpy as np
# import pandas as pd
#
# # Define the path to your dataset
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dataset_path = Path('../datasets/cats_vs_dogs')
#
#
# # Define your dataset class
# class CatsAndDogsDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = Path(root_dir)
#         self.transform = transform
#         self.images = [p for p in self.root_dir.glob('**/*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
#
#     def __len__(self):
#         return len(self.images)
#
#     def __getitem__(self, idx):
#         image_path = self.images[idx]
#         image = Image.open(image_path).convert('RGB')  # Convert the image to RGB
#
#         if self.transform:
#             image = self.transform(image)  # Apply the transform
#
#         label = 1 if image_path.parent.name == 'Dog' else 0
#         return image, label
#
#
# # Define the transform
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Resize to the input size of ResNet
#     transforms.ToTensor(),          # Convert image to a float tensor and scale to [0, 1]
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
#
# # Instantiate the dataset with the transform
# dataset = CatsAndDogsDataset(root_dir=dataset_path, transform=transform)
#
# # Use a DataLoader to handle batching
# dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
#
# # Load a pre-trained ResNet-50 model and remove the final fully connected layer
# model = models.resnet50(pretrained=True).to(device)
# feature_extractor = torch.nn.Sequential(*list(model.children())[:-1]).to(device)
# feature_extractor.eval()  # Set to evaluation mode
#
# # Disable gradient computation since we only need forward pass
# embeddings_list = []
# labels_list = []
#
# with torch.no_grad():
#     for images, labels in dataloader:
#         images = images.to(device)
#         labels = labels.to(device)
#
#         # Forward pass to get the embeddings
#         batch_embeddings = feature_extractor(images)
#
#         # Reshape embeddings to flatten them out
#         batch_embeddings = batch_embeddings.view(batch_embeddings.size(0), -1)
#
#         # Move the embeddings and labels to CPU and convert them to numpy arrays
#         embeddings_list.append(batch_embeddings.cpu().numpy())
#         labels_list.extend(labels.cpu().numpy())
#
# # Concatenate all embeddings and labels into a single NumPy array
# all_embeddings = np.concatenate(embeddings_list, axis=0)
# all_labels = np.array(labels_list).reshape(-1, 1)  # Reshape labels to be a column vector
#
# # Combine the embeddings and labels into one array
# final_data = np.hstack((all_embeddings, all_labels))
#
# # Create a DataFrame and save it to a CSV file
# column_names = [f'feature_{i + 1}' for i in range(final_data.shape[1] - 1)] + ['label']
# df = pd.DataFrame(final_data, columns=column_names)
# df.to_csv('../datasets/cats_vs_dogs/embeddings.csv', index=False)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from code.models.mlp import MLP
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import ConvergenceWarning
import time
import numpy as np
import math
from deap import gp, creator, base, tools
import operator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
df = pd.read_csv('../datasets/cats_vs_dogs/embeddings.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
pca = PCA(n_components=0.95)  # keeps 95% of variance
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(X_train_pca.shape, X_test_pca.shape)


# ======================================================================================================================
# Classic MLP
# ======================================================================================================================

# Create an instance of MLPClassifier
mlp = LogisticRegression(max_iter=1, warm_start=True)

train_log_losses = []
test_log_losses = []

# Number of iterations
n_iterations = 300

time_list = []
for i in range(n_iterations):
    # Train the model for one iteration
    start_time = time.time()
    mlp.fit(X_train_pca, y_train)
    time_list.append(time.time() - start_time)

    # Get probabilities for train and test sets
    train_probs = mlp.predict_proba(X_train_pca)
    test_probs = mlp.predict_proba(X_test_pca)

    # Calculate log loss for train and test sets
    train_loss = log_loss(y_train, train_probs)
    test_loss = log_loss(y_test, test_probs)

    # Append losses to lists
    train_log_losses.append(train_loss)
    test_log_losses.append(test_loss)

# Plotting the log losses over iterations
plt.figure(figsize=(10, 6))
plt.plot(train_log_losses, label='Train Log Loss')
plt.plot(test_log_losses, label='Test Log Loss')
plt.xlabel('Iteration')
plt.ylabel('Log Loss')
plt.title('Train and Test Log Loss over Iterations')
plt.legend()
plt.show()

best_test_loss = min(test_log_losses)
best_indexes = []
total_times_up_to_best = []
for index, loss in enumerate(test_log_losses):
    if loss == best_test_loss:
        best_indexes.append(index)
        total_time_up_to_index = sum(time_list[:index + 1])
        total_times_up_to_best.append(total_time_up_to_index)
print("Best Test Loss:", best_test_loss)
print("Indexes of Best Test Loss:", best_indexes)
print("Total Times up to these iterations (seconds):", total_times_up_to_best)

threshold = 0
accuracy_lst, precision_lst, recall_lst, f1_lst = [], [], [], []
while threshold <= 100:
    y_prob = mlp.predict_proba(X_test_pca)[:, 1]
    y_pred = (y_prob > threshold).astype(int)
    accuracy_lst.append((accuracy_score(y_test, y_pred), threshold))
    precision_lst.append((precision_score(y_test, y_pred, zero_division=0), threshold))
    recall_lst.append((recall_score(y_test, y_pred), threshold))
    f1_lst.append((f1_score(y_test, y_pred), threshold))
    threshold += 0.01
max_f1_score = max(f1_lst, key=lambda x: x[0])[0]
best_thresholds = [threshold for f1_score, threshold in f1_lst if f1_score == max_f1_score]
print("Classic MLP:")
for th in best_thresholds:
    index = [i for i, (f1, threshold) in enumerate(f1_lst) if threshold == th][0]
    print(f"Threshold={th:.2f}, Accuracy={accuracy_lst[index][0]:.4f}, "
          f"Precision={precision_lst[index][0]:.4f}, Recall={recall_lst[index][0]:.4f}, "
          f"F1-score={f1_lst[index][0]:.4f}\n")

# ======================================================================================================================
# MLP with GA
# ======================================================================================================================

model = MLP(hidden_layer_sizes=(X_train_pca.shape[1], 1), max_iter=20000)
model.fit(X_train_pca, y_train, check_test_statistic=True, X_test=X_test_pca, y_test=y_test)

iterations = list(range(1, len(model._errors) + 1))
plt.plot(iterations, model._errors, label='Train Loss')
plt.plot(iterations, model._errors_test, label='Test Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss Over Iterations')
plt.legend()
plt.show()

best_test_loss = min(model._errors_test)
best_indexes = [i for i, loss in enumerate(model._errors_test) if loss == best_test_loss]
total_times_up_to_best = [sum(model._times[:i + 1]) for i in best_indexes]
print("Best Test Loss:", best_test_loss)
print("Indexes of Best Test Loss:", best_indexes)
print("Total Times up to these iterations (seconds):", total_times_up_to_best)

threshold = 0
accuracy_lst, precision_lst, recall_lst, f1_lst = [], [], [], []
while threshold <= 100:
    y_pred = model.predict(X_test_pca, threshold=threshold)
    accuracy_lst.append((accuracy_score(y_test, y_pred), threshold))
    precision_lst.append((precision_score(y_test, y_pred, zero_division=0), threshold))
    recall_lst.append((recall_score(y_test, y_pred), threshold))
    f1_lst.append((f1_score(y_test, y_pred), threshold))
    threshold += 0.01
max_f1_score = max(f1_lst, key=lambda x: x[0])[0]
best_thresholds = [threshold for f1_score, threshold in f1_lst if f1_score == max_f1_score]
print("MLP with GA:")
for th in best_thresholds:
    index = [i for i, (f1, threshold) in enumerate(f1_lst) if threshold == th][0]
    print(f"Threshold={th:.2f}, Accuracy={accuracy_lst[index][0]:.4f}, "
          f"Precision={precision_lst[index][0]:.4f}, Recall={recall_lst[index][0]:.4f}, "
          f"F1-score={f1_lst[index][0]:.4f}\n")


# ======================================================================================================================
# (1 + lambda) - EA with GP
# ======================================================================================================================

def sigmoid(x):
    x_clipped = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x_clipped))


def evalSymbReg(individual, X, y):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the target values y
    predictions = np.array([sigmoid(func(*record)) for record in X])
    return log_loss(y, predictions),


def plot_losses(train_losses, test_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Generations')
    plt.ylabel('Loss')
    plt.title('Evolution of Train and Test Loss')
    plt.legend()
    plt.show()


def safe_div(x, y):
    return x / y if y != 0 else 1


pset = gp.PrimitiveSet("MAIN", X_train_pca.shape[1])
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(safe_div, 2)
pset.addPrimitive(min, 2)
pset.addPrimitive(max, 2)
pset.addPrimitive(math.hypot, 2)
pset.addPrimitive(np.logaddexp, 2)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=9, max_=9)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

toolbox.register("evaluate", evalSymbReg)
toolbox.register("select", tools.selTournament, tournsize=1)
toolbox.register("mutate", gp.mutNodeReplacement, pset=pset)

train_losses = []
test_losses = []
time_list = []

# Step 1: Initialization
champion = toolbox.individual()
fitness = evalSymbReg(champion, X_train_pca, y_train)
champion.fitness.values = fitness

# Parameters
λ = 2  # You can choose a different λ if you like
max_generations = 10  # You can choose a different number of generations

# Step 2: Evolutionary loop
for gen in range(max_generations):
    # Step 2.1: Generate λ candidates by mutating the current champion
    start_time = time.time()
    candidates = [toolbox.clone(champion) for _ in range(1 + λ)]
    for candidate in candidates:
        toolbox.mutate(candidate)
        del candidate.fitness.values

    # Step 2.2: Evaluate candidates
    for candidate in candidates:
        candidate.fitness.values = evalSymbReg(candidate, X_train_pca, y_train)

    # Step 2.3: Select the best individual
    candidates.append(champion)
    champion = tools.selBest(candidates, 1)[0]
    time_list.append(time.time() - start_time)

    # Calculate and store training loss
    train_loss = evalSymbReg(champion, X_train_pca, y_train)[0]
    train_losses.append(train_loss)

    # Calculate and store test loss
    test_loss = evalSymbReg(champion, X_test_pca, y_test)[0]
    test_losses.append(test_loss)

plot_losses(train_losses, test_losses)

best_test_loss = min(test_losses)
best_indexes = [i for i, loss in enumerate(test_losses) if loss == best_test_loss]
total_times_up_to_best = [sum(time_list[:i + 1]) for i in best_indexes]
print("Best Test Loss:", best_test_loss)
print("Indexes of Best Test Loss:", best_indexes)
print("Total Times up to these iterations (seconds):", total_times_up_to_best)


def make_predictions_with_threshold(individual, X, threshold=0.5):
    func = toolbox.compile(expr=individual)
    predictions_raw = np.array([func(*record) for record in X])
    predictions = sigmoid(predictions_raw)
    return (predictions > threshold).astype(int)


# Example thresholds from 0 to 1 (adjust step size as needed)
threshold = 0
accuracy_lst, precision_lst, recall_lst, f1_lst = [], [], [], []
while threshold <= 1.0:
    y_pred = make_predictions_with_threshold(champion, X_test_pca, threshold=threshold)
    accuracy_lst.append((accuracy_score(y_test, y_pred), threshold))
    precision_lst.append((precision_score(y_test, y_pred, zero_division=0), threshold))
    recall_lst.append((recall_score(y_test, y_pred), threshold))
    f1_lst.append((f1_score(y_test, y_pred), threshold))
    threshold += 0.01

max_f1_score = max(f1_lst, key=lambda x: x[0])[0]
best_thresholds = [threshold for f1_score, threshold in f1_lst if f1_score == max_f1_score]

print("(1 + lambda) - EA with GP:")
for th in best_thresholds:
    index = [i for i, (f1, threshold) in enumerate(f1_lst) if threshold == th][0]
    print(f"Threshold={th:.2f}, Accuracy={accuracy_lst[index][0]:.4f}, "
          f"Precision={precision_lst[index][0]:.4f}, Recall={recall_lst[index][0]:.4f}, "
          f"F1-score={f1_lst[index][0]:.4f}")

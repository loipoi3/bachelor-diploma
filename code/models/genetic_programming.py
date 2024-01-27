import numpy as np
import math
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import UndefinedMetricWarning
from deap import base, creator, tools, gp
import random
import warnings

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
# random.seed(42)
# np.random.seed(42)

# Load breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the genetic programming operators and functions
pset = gp.PrimitiveSet("MAIN", X_train_scaled.shape[1])
pset.addPrimitive(np.add, 2)
pset.addPrimitive(np.subtract, 2)
pset.addPrimitive(np.multiply, 2)
pset.addPrimitive(np.divide, 2)
pset.addPrimitive(np.power, 2)
pset.addPrimitive(np.maximum, 2)
pset.addPrimitive(np.minimum, 2)
pset.addPrimitive(np.greater, 2)

for idx in range(X_train_scaled.shape[1] - 1):
    pset.addTerminal(random.uniform(-2, 2), name=f'rand_val_{idx}')
pset.addTerminal(math.e, name='e_const')

creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genFull, pset=pset, min_=5, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# Define the evaluation function
def evaluate(individual):
    # Compile the expression
    func = gp.compile(individual, pset)

    # Make predictions on the training set
    probabilities = np.array([func(*x) for x in X_train])

    # Check for NaN or infinity values in probabilities
    if np.isnan(probabilities).any() or np.isinf(probabilities).any():
        # If NaN or infinity values are present, return a large loss
        return float('inf'),

    # Calculate log loss
    loss = log_loss(y_train, probabilities)

    return loss,


toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selBest)
toolbox.register("mutate", mutate_custom)

# Set up the algorithm parameters
population_size = 1  # Each iteration consists of one individual
n_mutations = 2  # Number of single-point mutations to be applied in each iteration
ngen = 10000

# Initialize the population
pop = toolbox.population(n=population_size)


# Function to extract and visualize the mathematical expression
def visualize_expression(individual):
    # Print the tree structure of the individual
    print("Mathematical Expression:")
    print(str(individual))
    print()


# Begin the evolution
for gen in range(ngen):
    # Clone the individual for mutation
    best_ind = toolbox.clone(pop[0])
    visualize_expression(best_ind)
    # Create n individuals by applying single-point mutations
    mutated_individuals = [toolbox.clone(best_ind) for _ in range(n_mutations)]
    for ind in mutated_individuals:
        toolbox.mutate(ind)
        # visualize_expression(ind)
        exit()
        # Evaluate the mutated individual
        fitness = toolbox.evaluate(ind)
        ind.fitness.values = fitness

    # Select the best individual among the mutated ones and the initial one
    best_ind = toolbox.select(mutated_individuals + [pop[0]], 1)[0]

    # Replace the current population with the selected individual
    pop[:] = [toolbox.clone(best_ind)]

# Get the best individual
best_ind = pop[0]

# Compile the best individual and make predictions on the test set
best_func = gp.compile(best_ind, pset)
test_predictions = [best_func(*x) for x in X_test]

# threshold = 0
# accuracy_lst, precision_lst, recall_lst, f1_lst = [], [], [], []
# while threshold <= 100:
#     y_pred = [0 if prob < threshold else 1 for prob in test_predictions]
#     accuracy_lst.append((accuracy_score(y_test, y_pred), threshold))
#     precision_lst.append((precision_score(y_test, y_pred), threshold))
#     recall_lst.append((recall_score(y_test, y_pred), threshold))
#     f1_lst.append((f1_score(y_test, y_pred), threshold))
#     threshold += 0.01
# best_f1_score, best_threshold = max(f1_lst, key=lambda x: x[0])
# best_f1_index = f1_lst.index((best_f1_score, best_threshold))
# best_accuracy = accuracy_lst[best_f1_index][0]
# best_precision = precision_lst[best_f1_index][0]
# best_recall = recall_lst[best_f1_index][0]
# print("(1+lambda)-EA with GP:")
# print(f"Best threshold={best_threshold}\nBest accuracy={best_accuracy}\nBest precision={best_precision}\n"
#       f"Best recall={best_recall}\nBest f1-score={best_f1_score}\n")

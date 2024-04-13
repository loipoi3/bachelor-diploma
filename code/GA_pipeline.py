import operator
from deap import gp, creator, base, tools
from matplotlib import pyplot as plt
from sklearn.metrics import log_loss
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import math

np.random.seed(42)


def sigmoid(x):
    x_clipped = np.clip(x, -500, 500)
    return 1 / (1 + math.exp(-x_clipped))


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


# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pset = gp.PrimitiveSet("MAIN", X_train_scaled.shape[1])
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
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=6, max_=6)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

toolbox.register("evaluate", evalSymbReg)
toolbox.register("select", tools.selTournament, tournsize=1)
toolbox.register("mutate", gp.mutNodeReplacement, pset=pset)


# Define the main algorithm using a (1+λ)-EA structure
def main():
    train_losses = []
    test_losses = []

    # Step 1: Initialization
    champion = toolbox.individual()
    fitness = evalSymbReg(champion, X_train_scaled, y_train)
    champion.fitness.values = fitness

    # Parameters
    λ = 2  # You can choose a different λ if you like
    max_generations = 5000  # You can choose a different number of generations

    # Step 2: Evolutionary loop
    for gen in range(max_generations):
        # Step 2.1: Generate λ candidates by mutating the current champion
        candidates = [toolbox.clone(champion) for _ in range(1 + λ)]
        for candidate in candidates:
            toolbox.mutate(candidate)
            del candidate.fitness.values

        # Step 2.2: Evaluate candidates
        for candidate in candidates:
            candidate.fitness.values = evalSymbReg(candidate, X_train_scaled, y_train)

        # Step 2.3: Select the best individual
        candidates.append(champion)
        champion = tools.selBest(candidates, 1)[0]

        # Calculate and store training loss
        train_loss = evalSymbReg(champion, X_train_scaled, y_train)[0]
        train_losses.append(train_loss)

        # Calculate and store test loss
        test_loss = evalSymbReg(champion, X_test_scaled, y_test)[0]
        test_losses.append(test_loss)

    return champion, train_losses, test_losses


# Finally, execute the main function
if __name__ == "__main__":
    champion, train_losses, test_losses = main()
    print(f"Fitness: {champion.fitness.values[0]}")
    plot_losses(train_losses, test_losses)

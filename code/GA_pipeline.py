# import numpy as np
# import math
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, f1_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.exceptions import UndefinedMetricWarning
# from deap import base, creator, tools, gp
# import random
# import warnings
#
# warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
#
#
# # random.seed(42)
# # np.random.seed(42)
#
#
# # Load breast cancer dataset
# data = load_breast_cancer()
# X, y = data.data, data.target
#
# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# # Define the genetic programming operators and functions
# pset = gp.PrimitiveSet("MAIN", X_train_scaled.shape[1])
# pset.addPrimitive(np.add, 2)
# pset.addPrimitive(np.subtract, 2)
# pset.addPrimitive(np.multiply, 2)
# pset.addPrimitive(np.divide, 2)
# pset.addPrimitive(np.power, 2)
# pset.addPrimitive(np.maximum, 2)
# pset.addPrimitive(np.minimum, 2)
#
# creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
# creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
#
# toolbox = base.Toolbox()
# toolbox.register("expr", gp.genFull, pset=pset, min_=3, max_=3)
# toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#
#
# # Define the evaluation function
# def evaluate(individual):
#     # Compile the expression
#     func = gp.compile(individual, pset)
#
#     # Make predictions on the training set
#     pred = np.array([func(*x) for x in X_train])
#     probabilities = 1 / (1 + np.exp(-pred))
#
#     # Check for NaN or infinity values in probabilities
#     if np.isnan(probabilities).any() or np.isinf(probabilities).any():
#         # If NaN or infinity values are present, return a large loss
#         return float('inf'),
#
#     # Calculate log loss
#     loss = log_loss(y_train, probabilities)
#
#     return loss,
#
#
# def mutate_custom(individual, pset):
#     # Randomly select a node in the individual
#     node_index = random.randrange(len(individual))
#
#     # If the selected node is not a terminal (function or operator)
#     if isinstance(individual[node_index], gp.Primitive):
#         # Randomly select a new operation from the primitive set
#         new_primitive = random.choice(list(pset.primitives.values())[0])
#
#         # Replace the primitive of the selected node with the new one
#         individual[node_index] = new_primitive
#
#     # If the selected node is a terminal (input variable)
#     elif isinstance(individual[node_index], gp.Terminal):
#         # Randomly select a new input variable (column)
#         new_terminal = random.choice(list(pset.terminals.values())[0])
#
#         # Replace the terminal of the selected node with the new one
#         # individual[node_index].name = new_terminal.name
#         # individual[node_index].value = new_terminal.value
#         individual[node_index] = new_terminal
#
#     return individual,
#
#
# toolbox.register("evaluate", evaluate)
# toolbox.register("select", tools.selBest)
# toolbox.register("mutate", mutate_custom, pset=pset)
#
# # Set up the algorithm parameters
# population_size = 1  # Each iteration consists of one individual
# n_mutations = 2  # Number of single-point mutations to be applied in each iteration
# ngen = 2
#
# # Initialize the population
# pop = toolbox.population(n=population_size)
#
#
# # Function to extract and visualize the mathematical expression
# def visualize_expression(individual):
#     # Print the tree structure of the individual
#     print("Mathematical Expression:")
#     print(str(individual))
#
#
# # Begin the evolution
# for gen in range(ngen):
#     # Clone the individual for mutation
#     best_ind = toolbox.clone(pop[0])
#     best_ind.fitness.values = toolbox.evaluate(best_ind)
#     print(f"Iter: {gen} Best ind:")
#     visualize_expression(best_ind)
#     print(f"Fitness: {best_ind.fitness.values}\n")
#
#     # Create n individuals by applying single-point mutations
#     mutated_individuals = [toolbox.clone(best_ind) for _ in range(n_mutations)]
#     for ind in mutated_individuals:
#         toolbox.mutate(ind)
#         print(f"Iter: {gen} Mut ind:")
#         visualize_expression(ind)
#
#         # Evaluate the mutated individual
#         fitness = toolbox.evaluate(ind)
#         print(f"Fitness: {fitness}\n")
#         ind.fitness.values = fitness
#
#     # Select the best individual among the mutated ones and the initial one
#     best_ind = toolbox.select(mutated_individuals + [pop[0]], 1)[0]
#     print(f"Iter: {gen} final Best ind:")
#     visualize_expression(best_ind)
#     print("=" * 100)
#     # Replace the current population with the selected individual
#     pop[:] = [toolbox.clone(best_ind)]
#
# # Get the best individual
# best_ind = pop[0]
#
# # Compile the best individual and make predictions on the test set
# best_func = gp.compile(best_ind, pset)
# test_predictions = np.array([best_func(*x) for x in X_test])
# test_predictions = 1 / (1 + np.exp(-test_predictions))


import operator
import random
import numpy as np
import math
from deap import base, creator, gp, tools

# Define the problem-specific parameters
# Here you should load your dataset and define the target column.
# For demonstration, I'll create a dummy dataset with random values.
np.random.seed(42)
X = np.random.rand(100, 5)  # 100 samples, 5 features
y = X[:, 0] + 2 * X[:, 1] ** 2 + np.random.rand(100) * 0.1  # Dummy target variable

# Define the primitive set for the tree-based GP
pset = gp.PrimitiveSet("MAIN", arity=5)  # Arity 5 for 5 input features
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(np.sin, 1)
pset.addPrimitive(np.cos, 1)

# Define a creator for the fitness (minimizing the error) and the individual (tree-based GP)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# Define the fitness function
def evaluate(individual, points, target):
    # Transform the tree expression to a callable function
    func = toolbox.compile(expr=individual)
    # Calculate the mean squared error between the predicted and true values
    sqerrors = ((func(*points[i]) - target[i])**2 for i in range(len(points)))
    return (math.fsum(sqerrors) / len(points),)

toolbox.register("evaluate", evaluate, points=X, target=y)
toolbox.register("select", tools.selBest)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Parameters
lambda_ = 5
mu = 1
ngen = 10  # Number of generations

# The (1+λ)-EA Algorithm, without crossover
def one_plus_lambda_ea():
    # Initialize the population with 1 individual
    population = toolbox.population(n=mu)
    halloffame = tools.HallOfFame(1)

    # Evaluate the initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Update the hall of fame with the initial population
    halloffame.update(population)

    # Begin the evolution
    for gen in range(ngen):
        # A new generation
        for _ in range(lambda_):
            # Clone the best individual and apply mutation
            mutant = toolbox.clone(halloffame.items[0])  # Access the hall of fame's items directly
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

            # Evaluate the new individuals
            fit = toolbox.evaluate(mutant)
            mutant.fitness.values = fit
            # If the mutant is better than the current best, replace it
            if mutant.fitness > population[0].fitness:
                population[0] = mutant
                halloffame.update(population)

    return population, halloffame

if __name__ == "__main__":
    final_pop, hof = one_plus_lambda_ea()
    best_expr = hof[0]
    print(best_expr)
    print(toolbox.evaluate(best_expr))

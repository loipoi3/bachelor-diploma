import time
import numpy as np
import operator
import math
from deap import gp, creator, base, tools
from sklearn.metrics import log_loss


class GeneticAlgorithmModel:
    """
    Class implementation of a genetic algorithm model using DEAP library for symbolic regression with
    logistic loss minimization.

    Attributes:
        X_train (ndarray): Training feature data.
        y_train (ndarray): Training labels.
        X_test (ndarray): Test feature data.
        y_test (ndarray): Test labels.
        pset (PrimitiveSet): Set of primitive operations for the genetic program.
        toolbox (base.Toolbox): DEAP Toolbox with genetic operators.
    """

    def __init__(self, X_train, y_train, X_test, y_test):
        """
        Initializes the genetic algorithm model with training and test data.

        Parameters:
            X_train (ndarray): Training feature data.
            y_train (ndarray): Training labels.
            X_test (ndarray): Test feature data.
            y_test (ndarray): Test labels.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # Define a primitive set for symbolic regression with the number of input variables equal to X_train columns
        self.pset = gp.PrimitiveSet("MAIN", X_train.shape[1])
        self._setup_primitives()

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self._setup_toolbox()

    def _setup_primitives(self):
        """Adds basic arithmetic and mathematical operations to the primitive set."""
        self.pset.addPrimitive(operator.add, 2)
        self.pset.addPrimitive(operator.sub, 2)
        self.pset.addPrimitive(operator.mul, 2)
        self.pset.addPrimitive(self._safe_div, 2)
        self.pset.addPrimitive(min, 2)
        self.pset.addPrimitive(max, 2)
        self.pset.addPrimitive(math.hypot, 2)
        self.pset.addPrimitive(np.logaddexp, 2)

    def _setup_toolbox(self):
        """Sets up the DEAP toolbox for genetic programming."""
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=6, max_=6)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        self.toolbox.register("evaluate", self._evalSymbReg)
        self.toolbox.register("select", tools.selTournament, tournsize=1)
        self.toolbox.register("mutate", gp.mutNodeReplacement, pset=self.pset)

    def _safe_div(self, x, y):
        """Performs safe division to prevent division by zero."""
        return x / y if y != 0 else 1

    def _sigmoid(self, x):
        """Applies the sigmoid function to clip and transform input to a range between 0 and 1."""
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))

    def _evalSymbReg(self, individual, X, y):
        """
        Evaluates an individual's fitness using logistic loss.

        Parameters:
            individual: The individual to evaluate.
            X (ndarray): Input data.
            y (ndarray): Target labels.

        Returns:
            tuple: Contains one element, the logistic loss of the individual.
        """
        func = self.toolbox.compile(expr=individual)
        predictions = np.array([self._sigmoid(func(*record)) for record in X])
        return log_loss(y, predictions),

    def run(self, lambd: int, max_generations: int) -> tuple:
        """
        Executes the genetic algorithm.

        Parameters:
            lambd (int): The number of mutations to generate per generation.
            max_generations (int): The number of generations to run the algorithm.

        Returns:
            tuple: Contains the best individual, training losses, test losses, and timing information.
        """
        train_losses, test_losses, time_list = [], [], []

        champion = self.toolbox.individual()
        champion.fitness.values = self._evalSymbReg(champion, self.X_train, self.y_train)

        for gen in range(max_generations):
            start_time = time.time()
            candidates = [self.toolbox.clone(champion) for _ in range(1 + lambd)]
            for candidate in candidates:
                self.toolbox.mutate(candidate)
                del candidate.fitness.values

            for candidate in candidates:
                candidate.fitness.values = self._evalSymbReg(candidate, self.X_train, self.y_train)

            candidates.append(champion)
            champion = tools.selBest(candidates, 1)[0]
            time_list.append(time.time() - start_time)

            train_loss = self._evalSymbReg(champion, self.X_train, self.y_train)[0]
            train_losses.append(train_loss)

            test_loss = self._evalSymbReg(champion, self.X_test, self.y_test)[0]
            test_losses.append(test_loss)

        return champion, train_losses, test_losses, time_list

    def make_predictions_with_threshold(self, individual, X, threshold: float) -> int:
        """
        Makes predictions using the compiled expression of an individual, with a given threshold for classification.

        Parameters:
            individual: The individual whose compiled expression is used for making predictions.
            X (ndarray): Input data.
            threshold (float): Threshold for converting probability to class labels.

        Returns:
            ndarray: Predicted class labels.
        """
        func = self.toolbox.compile(expr=individual)
        predictions_raw = np.array([func(*record) for record in X])
        predictions = self._sigmoid(predictions_raw)
        return (predictions > threshold).astype(int)

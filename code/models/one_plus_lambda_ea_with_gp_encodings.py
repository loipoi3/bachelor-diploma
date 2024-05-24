import time
import numpy as np
import operator
import math
from deap import gp, creator, base, tools
from sklearn.metrics import log_loss
from scipy.special import softmax
import random
import pickle
import os


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

    def __init__(self, X_train, y_train, X_test, y_test, tree_depth, primitive_set, terminal_set, num_classes=1):
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
        self.tree_depth = tree_depth
        self._num_classes = num_classes
        self._primitive_set = primitive_set
        self._terminal_set = terminal_set

        # Define a primitive set for symbolic regression with the number of input variables equal to X_train columns
        self.pset = gp.PrimitiveSet("MAIN", X_train.shape[1])
        self._setup_primitives()

        if hasattr(creator, 'FitnessMin'):
            del creator.FitnessMin
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if hasattr(creator, 'Individual'):
            del creator.Individual
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self._setup_toolbox()

    def _setup_primitives(self):
        """Adds basic arithmetic and mathematical operations to the primitive set."""
        if "add" in self._primitive_set:
            self.pset.addPrimitive(operator.add, 2)
        if "sub" in self._primitive_set:
            self.pset.addPrimitive(operator.sub, 2)
        if "mul" in self._primitive_set:
            self.pset.addPrimitive(operator.mul, 2)
        if "_safe_div" in self._primitive_set:
            self.pset.addPrimitive(self._safe_div, 2)
        if "min" in self._primitive_set:
            self.pset.addPrimitive(min, 2)
        if "max" in self._primitive_set:
            self.pset.addPrimitive(max, 2)
        if "hypot" in self._primitive_set:
            self.pset.addPrimitive(math.hypot, 2)
        if "logaddexp" in self._primitive_set:
            self.pset.addPrimitive(np.logaddexp, 2)
        if "_safe_atan2" in self._primitive_set:
            self.pset.addPrimitive(self._safe_atan2, 2)
        if "_float_lt" in self._primitive_set:
            self.pset.addPrimitive(self._float_lt, 2)
        if "_float_gt" in self._primitive_set:
            self.pset.addPrimitive(self._float_gt, 2)
        if "_float_ge" in self._primitive_set:
            self.pset.addPrimitive(self._float_ge, 2)
        if "_float_le" in self._primitive_set:
            self.pset.addPrimitive(self._float_le, 2)
        if "_safe_fmod" in self._primitive_set:
            self.pset.addPrimitive(self._safe_fmod, 2)
        if "Constant_0" in self._terminal_set:
            self.pset.addTerminal(0, "Constant_0")
        if "Constant_1" in self._terminal_set:
            self.pset.addTerminal(1, "Constant_1")
        if "Constant_minus_1" in self._terminal_set:
            self.pset.addTerminal(-1, "Constant_minus_1")
        if "Pi" in self._terminal_set:
            self.pset.addTerminal(math.pi, "Pi")
        if "E" in self._terminal_set:
            self.pset.addTerminal(math.e, "E")

    def _setup_toolbox(self):
        """Sets up the DEAP toolbox for genetic programming."""
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=self.tree_depth, max_=self.tree_depth)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        self.toolbox.register("evaluate", self._evalSymbReg)
        self.toolbox.register("select", tools.selTournament, tournsize=1)
        self.toolbox.register("mutate", gp.mutNodeReplacement, pset=self.pset)

    def _safe_div(self, x, y):
        """Performs safe division to prevent division by zero and handle overflow."""
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            result = np.divide(x, y)
            if np.isnan(result) or np.isinf(result):
                return 1  # or another appropriate value depending on your application needs
            return result

    def _safe_atan2(self, y, x):
        # Check for the special case where both inputs are zero
        if y == 0 and x == 0:
            return 1  # Return a small non-zero angle in radians instead of zero
        # Normal operation
        return math.atan2(y, x)

    def _float_lt(self, a, b):
        return 1.0 if operator.lt(a, b) else 0.0

    def _float_gt(self, a, b):
        return 1.0 if operator.gt(a, b) else 0.0

    def _float_ge(self, a, b):
        return 1.0 if operator.ge(a, b) else 0.0

    def _float_le(self, a, b):
        return 1.0 if operator.le(a, b) else 0.0

    def _safe_fmod(self, x, y):
        if y == 0:
            return 1.0  # Provide a sensible default, or handle the error as needed
        return math.fmod(x, y)

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
        if self._num_classes == 1:
            func = self.toolbox.compile(expr=individual)
            # vectorized_func = np.vectorize(func)
            # predictions = self._sigmoid(vectorized_func(*np.hsplit(X, X.shape[1])))
            predictions = np.array([self._sigmoid(func(*record)) for record in X])
        else:
            funcs = [self.toolbox.compile(expr=tree) for tree in individual["ind"]]
            predictions = np.array([[func(*record) for func in funcs] for record in X])
            predictions = softmax(predictions, axis=1)
        return log_loss(y, predictions),

    def run(self, lambd: int, max_generations: int, save_checkpoint_path, start_checkpoint: str = "",
            save_checkpoint: bool = False, mlp_time=0) -> tuple:
        """
        Executes the genetic algorithm.

        Parameters:
            lambd (int): The number of mutations to generate per generation.
            max_generations (int): The number of generations to run the algorithm.

        Returns:
            tuple: Contains the best individual, training losses, test losses, and timing information.
        """
        if start_checkpoint != "":
            state = self._load_checkpoint(start_checkpoint)
            champion = state['champion']
            start_generation = state['generation'] + 1
            train_losses = state['train_losses']
            test_losses = state['test_losses']
            time_list = state['time_list']
        else:
            train_losses, test_losses, time_list = [], [], []

            if self._num_classes == 1:
                champion = self.toolbox.individual()
                champion.fitness.values = self._evalSymbReg(champion, self.X_train, self.y_train)
            else:
                champion = {"ind": [self.toolbox.individual() for _ in range(self._num_classes)],
                            "fitness": {"values": None}}
                champion_fitness = self._evalSymbReg(champion, self.X_train, self.y_train)
                champion["fitness"]["values"] = champion_fitness[0]

            start_generation = 0

        for gen in range(start_generation, max_generations):
            # print(gen)
        # while sum(time_list) < mlp_time:
            start_time = time.time()
            if self._num_classes == 1:
                candidates = [self.toolbox.clone(champion) for _ in range(1 + lambd)]
                for candidate in candidates:
                    self.toolbox.mutate(candidate)
                    del candidate.fitness.values
            else:
                candidates = [
                    {"ind": [self.toolbox.clone(tree) for tree in champion["ind"]], "fitness": {"values": None}} for _
                    in range(1 + lambd)]
                for candidate in candidates:
                    selected_tree = random.choice(candidate["ind"])
                    self.toolbox.mutate(selected_tree)
                    del selected_tree.fitness.values

            if self._num_classes == 1:
                for candidate in candidates:
                    candidate.fitness.values = self._evalSymbReg(candidate, self.X_train, self.y_train)
            else:
                for candidate in candidates:
                    candidate["fitness"]["values"] = self._evalSymbReg(candidate, self.X_train, self.y_train)[0]

            candidates.append(champion)
            if self._num_classes == 1:
                champion = tools.selBest(candidates, 1)[0]
            else:
                sorted_list = sorted(candidates, key=lambda x: x["fitness"]["values"])
                champion = sorted_list[0]
            time_list.append(time.time() - start_time)

            train_loss = self._evalSymbReg(champion, self.X_train, self.y_train)[0]
            train_losses.append(train_loss)

            test_loss = self._evalSymbReg(champion, self.X_test, self.y_test)[0]
            test_losses.append(test_loss)

            if save_checkpoint:
                self._save_checkpoint(champion, gen, train_losses, test_losses, time_list, save_checkpoint_path)
        for i in range(1, len(time_list) + 1):
            if i % 5000 == 0:
                print("Iter: ", i)
                print("Time: ", sum(time_list[:i]))
                print("train_loss: ", train_losses[i-1])
                print("test_loss: ", test_losses[i-1])
        print("Time list: ", time_list)
        print("Train loss list: ", train_losses)
        print("Test loss list: ", test_losses)
        return champion, train_losses, test_losses, time_list

    def make_predictions_with_threshold(self, individual, X, threshold: float = 0.5) -> int:
        """
        Makes predictions using the compiled expression of an individual, with a given threshold for classification.

        Parameters:
            individual: The individual whose compiled expression is used for making predictions.
            X (ndarray): Input data.
            threshold (float): Threshold for converting probability to class labels.

        Returns:
            ndarray: Predicted class labels.
        """
        if self._num_classes == 1:
            func = self.toolbox.compile(expr=individual)
            predictions_raw = np.array([func(*record) for record in X])
            predictions = self._sigmoid(predictions_raw)
            return (predictions > threshold).astype(int)
        else:
            funcs = [self.toolbox.compile(expr=tree) for tree in individual["ind"]]
            predictions_raw = np.array([[func(*record) for func in funcs] for record in X])
            predictions = softmax(predictions_raw, axis=1)
            return np.argmax(predictions, axis=1)

    def _save_checkpoint(self, champion, generation, train_losses, test_losses, time_list, directory):
        """
        Saves the current state of the genetic algorithm run to a file.

        Parameters:
            champion: The best individual so far.
            generation: Current generation number.
            train_losses: List of training losses.
            test_losses: List of test losses.
            time_list: List of iteration times.
            directory: The directory to save checkpoints.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        filename = os.path.join(directory, f"checkpoint_gen_{generation}.pkl")
        with open(filename, 'wb') as cp_file:
            pickle.dump({
                "champion": champion,
                "generation": generation,
                "train_losses": train_losses,
                "test_losses": test_losses,
                "time_list": time_list
            }, cp_file)

    def _load_checkpoint(self, filename):
        """
        Loads a genetic algorithm checkpoint from a file.

        Parameters:
            filename: Path to the checkpoint file.

        Returns:
            Loaded state dictionary.
        """
        with open(filename, 'rb') as cp_file:
            state = pickle.load(cp_file)
        return state

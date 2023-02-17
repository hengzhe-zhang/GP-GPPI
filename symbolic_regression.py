import numpy as np
from deap import base, creator, gp, tools, algorithms
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, make_scorer


class GPIndividualEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, pset, individual):
        self.pset = pset
        self.individual = individual
        self.func = gp.compile(self.individual, self.pset)

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.array([self.func(*x) for x in X])


class SymbolicRegressorGPPI(BaseEstimator, RegressorMixin):
    """Scikit-learn compatible class for symbolic regression using DEAP."""

    def __init__(self, n_generations=100, pop_size=100, crossover_prob=0.9, mutation_prob=0.1, n_selected_features=None,
                 verbose=False):
        self.n_generations = n_generations
        self.pop_size = pop_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.n_selected_features = n_selected_features
        self.verbose = verbose
        self.toolbox = base.Toolbox()

    def evaluate_individual(self, individual):
        func = gp.compile(individual, self.pset)
        y_pred = []
        for row in self.X:
            try:
                y_pred.append(func(*row))
            except:
                y_pred.append(np.nan)
        return np.mean((self.y - y_pred) ** 2),

    def fit(self, X, y):
        self.X = X
        self.y = y

        # Define primitive set
        pset = gp.PrimitiveSet("MAIN", X.shape[1])
        self.add_functions_to_pset(pset)
        pset.addEphemeralConstant("rand101", lambda: np.random.randint(-1, 2))
        self.pset = pset

        # Define individual and population
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
        self.define_toolbox()

        # Initialize population
        pop = self.toolbox.population(n=self.pop_size)

        # Define the statistics
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(lambda ind: ind.height)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)

        # Store top 10 individuals
        hof = tools.HallOfFame(10)

        # Run GP algorithm for full set of features
        algorithms.eaSimple(pop, self.toolbox, cxpb=self.crossover_prob, mutpb=self.mutation_prob,
                            ngen=self.n_generations, stats=mstats, halloffame=hof, verbose=self.verbose)

        # Compile the top individuals and wrap them as scikit-learn regressors
        regressors = []
        for i, func in enumerate(hof):
            estimator = GPIndividualEstimator(pset, func)
            regressors.append(estimator)

        # Calculate permutation importance scores based on top 10 individuals
        scores = permutation_importance(regressors[0], self.X, self.y, scoring=make_scorer(r2_score),
                                        n_repeats=10, random_state=0)
        feature_importance = scores.importances_mean
        for func in regressors[1:]:
            scores = permutation_importance(func, self.X, self.y, scoring=make_scorer(r2_score),
                                            n_repeats=10, random_state=0)
            feature_importance += scores.importances_mean
        feature_importance /= len(regressors)
        if self.verbose:
            print('Feature importance score', feature_importance)

        # Select features with high permutation importance
        if self.n_selected_features is not None:
            indices = np.argsort(feature_importance)[::-1][:self.n_selected_features]
        else:
            indices = np.where(feature_importance > 0)[0]
        self.selected_indices = indices
        self.X = self.X[:, self.selected_indices]

        # Update the primitive set and population
        self.pset = gp.PrimitiveSet("MAIN", len(self.selected_indices))
        self.add_functions_to_pset(self.pset)
        self.pset.addEphemeralConstant("new_rand101", lambda: np.random.randint(-1, 2))
        self.define_toolbox()

        # Initialize population
        pop = self.toolbox.population(n=self.pop_size)

        # Run GP algorithm for selected features
        algorithms.eaSimple(pop, self.toolbox, cxpb=self.crossover_prob, mutpb=self.mutation_prob,
                            ngen=self.n_generations, stats=mstats, halloffame=hof, verbose=self.verbose)

        # Compile the best individual and store it
        top = tools.selBest(hof, k=1)
        self.best_individual_ = top[0]
        self.best_func_ = gp.compile(self.best_individual_, self.pset)
        return self

    def add_functions_to_pset(self, pset):
        pset.addPrimitive(np.add, 2)
        pset.addPrimitive(np.subtract, 2)
        pset.addPrimitive(np.multiply, 2)
        pset.addPrimitive(np.negative, 1)
        pset.addPrimitive(np.sin, 1)
        pset.addPrimitive(np.cos, 1)
        pset.addPrimitive(np.tan, 1)

    def define_toolbox(self):
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=3)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.evaluate_individual)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def predict(self, X):
        y_pred = []
        for row in X[:, self.selected_indices]:
            try:
                y_pred.append(self.best_func_(*row))
            except:
                y_pred.append(np.nan)
        return y_pred

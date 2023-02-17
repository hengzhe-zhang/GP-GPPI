# GP-GPPI: Symbolic Regression with Feature Selection using GP 

This is an implementation of symbolic regression with feature selection using GP based on the paper by Chen, Qi, Mengjie Zhang, and Bing Xue, "Feature Selection to Improve Generalization of Genetic Programming for High-Dimensional Symbolic Regression" published in IEEE Transactions on Evolutionary Computation.

**Disclaimer**: This package is not the original implementation of the method described in the paper "Feature Selection to Improve Generalization of Genetic Programming for High-Dimensional Symbolic Regression" by Chen, Qi, Mengjie Zhang, and Bing Xue. It is based on the paper and implements the described method using the `DEAP` and `scikit-learn` libraries. Please note that the results obtained using this package may differ from the results reported in the paper, as the package may not exactly reproduce the original implementation. Users are advised to use this package with caution and carefully validate their results before drawing conclusions.

## Problem

Symbolic regression is a type of regression analysis that finds a mathematical expression that best fits a set of data. The goal is to find an equation that describes the relationship between the input variables and the output variable. However, in high-dimensional data, genetic programming (GP) typically could not generalize well. Feature selection can potentially improve the efficiency of learning algorithms and enhance the generalization ability.

## Solution

This implementation proposes a feature selection method based on permutation to select features for high-dimensional symbolic regression using GP. In the first stage, a full GP algorithm is performed on all features. Then, features with high permutation importance values are selected as subsets. In the second stage, a full GP algorithm is performed on the selected features.

## Usage

The implementation provides a `SymbolicRegressorDEAP` class that can be used like any other scikit-learn estimator. This class has hyperparameters that can be set, such as the population size, the number of generations, and the selection and mutation operators. An example usage is shown below:

```python
from symbolic_regression import SymbolicRegressorGPPI

regressor = SymbolicRegressorGPPI(n_generations=20, verbose=True)
regressor.fit(X, y)
```

The `fit()` method will train the symbolic regressor on the provided features `X` and target `y`. After fitting, the `regressor` object can be used to make predictions on new data using the `predict()` method.

## References

- Chen, Qi, Mengjie Zhang, and Bing Xue. "Feature Selection to Improve Generalization of Genetic Programming for High-Dimensional Symbolic Regression." IEEE Transactions on Evolutionary Computation 21.5 (2017): 792-806.
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from symbolic_regression import SymbolicRegressorGPPI

# Load Boston housing dataset
boston = load_boston()
X = boston.data
y = boston.target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create instance of SymbolicRegressorDEAP and fit to training data
sr = SymbolicRegressorGPPI(n_generations=20, verbose=True)
sr.fit(X_train, y_train)

# Make predictions on test set
y_pred = sr.predict(X_test)

# Evaluate performance using R-squared score
r2 = r2_score(y_test, y_pred)
print(f"R-squared score: {r2:.2f}")

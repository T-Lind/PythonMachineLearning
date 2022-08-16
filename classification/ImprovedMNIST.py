import numpy as np
from numpy import ndarray
from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_curve, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Get the MNIST dataset
from sklearn.utils import Bunch


mnist: Bunch = fetch_openml('mnist_784', as_frame=False)

# Get the data for MNIST - X is the 70,000 images, y is the # features per image (pixels),
# each pixel is an intensity feature
X: ndarray
y: ndarray
X, y = mnist["data"], mnist["target"]
y = y.astype(np.uint8)

# Split the train and test data and scale it
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

k_clf = KNeighborsClassifier()
k_clf.fit(X_train[:1000], y_train[:1000])

# All the options to use
param_grid = [{'algorithm': ["ball_tree"], 'leaf_size': [5, 10, 30, 50], 'p': [1, 2, 3, 4], 'n_jobs':[None, -1]}]

# Use a grid search to find the right hyperparameters
grid_search = RandomizedSearchCV(k_clf, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(X_train[:2000], y_train[:2000])

print(grid_search.best_params_)

best_model = grid_search.best_estimator_

y_test_pred = best_model.predict(X_test)

print("Precision:", precision_score(y_test, y_test_pred, average="micro"))
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("Recall:", recall_score(y_test, y_test_pred, average="micro"))

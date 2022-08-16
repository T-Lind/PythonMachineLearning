import numpy as np
from numpy import ndarray
from sklearn.datasets import fetch_openml

from sklearn.metrics import accuracy_score
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
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


svm_clf = SVC(gamma="auto", random_state=42, probability=True, decision_function_shape='ovo')
svm_clf.fit(X_train[:1000], y_train[:1000])

# All the options to use
# param_grid = [{'gamma': ['auto'], 'decision_function_shape': ['ovo', 'ovr']}]
#
# # Use a grid search to find the right hyperparameters
# grid_search = RandomizedSearchCV(svm_clf, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
# grid_search.fit(X_train[:1000], y_train[:1000])
#
# print(grid_search.best_params_)
best_model = svm_clf

# best_model = grid_search.best_estimator_

print(best_model.predict([X[1]]))

y_test_pred = best_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_test_pred))


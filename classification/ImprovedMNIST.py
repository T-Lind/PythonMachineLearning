import numpy as np
from numpy import ndarray
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

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

# Use four classifiers and combine them with voting, then use an AdaBoostClassifier to boost the output
# k_clf = KNeighborsClassifier(weights="distance", algorithm="ball_tree")
log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()

stacking_clf = StackingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    cv=3
)

ada_clf = AdaBoostClassifier(stacking_clf, n_estimators=200, algorithm="SAMME", learning_rate=0.5)
ada_clf.fit(X_train[:1000], y_train[:1000])

best_model = ada_clf

y_test_pred = best_model.predict(X_test)
print("Precision:", precision_score(y_test, y_test_pred, average="weighted"))
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("Recall:", recall_score(y_test, y_test_pred, average="weighted"))

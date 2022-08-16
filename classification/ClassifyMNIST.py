import numpy as np
from numpy import ndarray
from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_curve
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


svm_clf = SVC(gamma="auto", random_state=42)
svm_clf.fit(X_train[:100], y_train[:100])
print(svm_clf.predict([X[1]]))

# y_train_pred = cross_val_predict(svm_clf, X_train, y_train, cv=3)
# conf_mx = confusion_matrix(y_train, y_train_pred)
#
# plt.matshow(conf_mx, cmap=plt.cm.gray)
# plt.show()

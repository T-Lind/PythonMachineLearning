import numpy as np
from numpy import ndarray
from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_curve

# Get the MNIST dataset
from sklearn.utils import Bunch

from support import *


def plot_digit(image_data):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")


mnist: Bunch = fetch_openml('mnist_784', as_frame=False)

# Get the data for MNIST - X is the 70,000 images, y is the # features per image (pixels),
# each pixel is an intensity feature
X: ndarray
y: ndarray
X, y = mnist["data"], mnist["target"]
y = y.astype(np.uint8)

# Load an example image and plot it
some_digit = X[0]  # Get the first image in the list
some_digit_image = some_digit.reshape(28, 28)  # Reshape it to a 2d array

plt.imshow(some_digit_image, cmap="binary")  # Display image in binary(grayscale) color
plt.axis("off")  # Turn the plot axis off
plt.show()
print(y[0])

# Split the train and test data
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Take out only the 5 labeled data
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

# Create a model to perform detections by using stochastic gradient descent
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

# Print the prediction of one example
print(sgd_clf.predict([some_digit]))

# Plot 100 numbers in a 10x10 grid
# plt.figure(figsize=(9, 9))
# for idx, image_data in enumerate(X[:100]):
#     plt.subplot(10, 10, idx + 1)
#     plot_digit(image_data)
# plt.subplots_adjust(wspace=0, hspace=0)
# plt.show()

# Get the predictions from all the data
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

# Print the number of true positives, negatives, false positives, negatives
print(confusion_matrix(y_train_5, y_train_pred))

# Print the percent it is right when classifying something as a 5
print(precision_score(y_train_5, y_train_pred))

# Print the percent of total 5s it gets
print(recall_score(y_train_5, y_train_pred))

# Try a RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")

y_scores_forest = y_probas_forest[:, 1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)

plt.plot(fpr_forest, tpr_forest)
plt.show()

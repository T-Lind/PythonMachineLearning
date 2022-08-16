import numpy as np
from numpy import ndarray
from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt

# Get the MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

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


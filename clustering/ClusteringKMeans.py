import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate blob data
from support import plot_decision_boundaries


def plot_clusters(X, y=None):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14, rotation=0)

blob_centers = np.array(
    [[0.2, 2.3],
     [-1.5, 2.3],
     [-2.8, 1.8],
     [-2.8, 2.8],
     [-2.8, 1.3]])
blob_std = np.array([0.2, 0.1, 0.1, 0.1, 0.1])

# Create the blob set
X, y = make_blobs(n_samples=2000, centers=blob_centers,
                  cluster_std=blob_std, random_state=7)

# Show the blobs
plt.figure(figsize=(8, 4))
plot_clusters(X)
plt.show()

k = 5
kmeans = KMeans(n_clusters=k)
y_pred = kmeans.fit_predict(X)

print(kmeans.cluster_centers_)

# Show the cluster boundaries
plt.figure(figsize=(8, 4))
plot_decision_boundaries(kmeans, X)
plt.show()

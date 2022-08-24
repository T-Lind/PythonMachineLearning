import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from joblib.numpy_pickle_utils import xrange
from pandas import DataFrame
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from matplotlib.colors import ListedColormap
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from sklearn.model_selection import train_test_split

from support import plot_decision_boundaries, DATASETS

# Get the housing.csv file from my datasets
housing: DataFrame = pd.read_csv(DATASETS + "housing.csv")
housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]


X = list(zip(housing["rooms_per_house"], housing["median_house_value"]))
X = X[:1000]

y = [x > 5 for x in housing["median_income"]]
y = y[:1000]

per_clf = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
per_clf.fit(X, y)

plt.plot(X)
plt.show()

print(per_clf.predict([[200, 100000]]))

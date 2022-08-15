import time

import pandas as pd
from pandas import DataFrame
from pandas.io.parsers import TextFileReader

from support import *

from matplotlib import pyplot as plt

# two_dim_data: TextFileReader | DataFrame = pd.read_csv(DATASETS + "simple-2d-data-train.csv")
#
# two_dim_data.plot(kind="scatter", x="Feature1", y="Feature2")

housing: TextFileReader | DataFrame = pd.read_csv(DATASETS + "housing.csv")
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

plt.show()

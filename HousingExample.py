from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from support import *

housing: DataFrame = pd.read_csv(DATASETS+"housing.csv")

# extra code – the next 5 lines define the default font sizes
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

housing.hist(bins=50, figsize=(12, 8))
plt.show()
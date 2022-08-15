import pandas as pd
from pandas import DataFrame
from pandas.io.parsers import TextFileReader

from support import *
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

winequality_red: TextFileReader | DataFrame = pd.read_csv(DATASETS + "winequality-red.csv", sep=";")

# Print the first 5 elements of the dataset
print(winequality_red.head(), "\n")

# Get a summary of each numerical attribute
print(winequality_red.describe())

# Generate the train and test data from the dataset using a shuffle split
train: DataFrame
test: DataFrame
train, test = train_test_split(winequality_red, "quality")

# Generate a histogram of the data. In this case it creates many, one for each feature. Automatically fed to matplotlib
train.hist(bins=50, figsize=(20, 15))
test.hist(bins=50, figsize=(20, 15))

plt.show()

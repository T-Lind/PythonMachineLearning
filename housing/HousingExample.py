import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from pandas.plotting import scatter_matrix
from support import *

# Get the housing.csv file from my datasets
housing: DataFrame = pd.read_csv(DATASETS+"housing.csv")

# define the default font sizes
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

# Create a plot of the housing data, with x being the longitude column and y being the latitude column
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
plt.show()

# Create a correlation matrix which defines the similarity between features - pos/neg correlation
corr_matrix: DataFrame = housing.corr()

# Print the correlation with median house value - median income is closest
print(corr_matrix["median_house_value"].sort_values(ascending=False))

# Show a 4x4 plot of the below four attributes against each other to show correlations easier
attributes: list = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.show()

# Create new attributes based on the previous data
housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"] = housing["population"] / housing["households"]

# Create the new correlation matrix
corr_matrix: DataFrame = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas import DataFrame
from scipy.sparse import csr_matrix

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

from sklearn import set_config


from support import *

set_config(display='text')

# Get the housing.csv file from my datasets
housing: DataFrame = pd.read_csv(DATASETS+"housing.csv")

# Create new attributes based on the previous data
housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"] = housing["population"] / housing["households"]

# Split the data
train: DataFrame
test: DataFrame
train, test = train_test_split(housing, test_size=0.2, random_state=42)

# Get rid of the house value from the train set and assign to housing
housing: DataFrame = train.drop("median_house_value", axis=1)
housing_labels = train["median_house_value"].copy()

# Find null rows in the dataset
null_rows_idx = housing.isnull().any(axis=1)
housing_option1 = housing.copy()
housing_option1.dropna(subset=["total_bedrooms"], inplace=True)  # option 1
housing_option1.loc[null_rows_idx].head()

# For null values in the dataset, replace with the median
imputer = SimpleImputer(strategy="median")

# Remove the ocean proximity column because it is not numerical (it's words, like close, far, etc.)
housing_num: DataFrame = housing.drop("ocean_proximity", axis=1)

# Apply the imputer to remove the null rows and assign it
imputer.fit(housing_num)
X = imputer.transform(housing_num)

# Convert the imputer result back to a DataFrame
housing_tr: DataFrame = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)

# Encode the ocean_proximity data
housing_cat: DataFrame = housing[["ocean_proximity"]]
cat_encoder = OneHotEncoder()
housing_cat_1hot: csr_matrix = cat_encoder.fit_transform(housing_cat)

# Construct a pipeline
num_pipeline: Pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
housing_num_tr = num_pipeline.fit_transform(housing_num)

# Get the attributes for the data and combine them
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

# Generate the prepared data
housing_prepared = full_pipeline.fit_transform(housing)

# Create predictions using a Decision Tree Regressor and automatically tweak hyperparameters using a Grid Search

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features':[2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)

grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)

cvres: dict = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


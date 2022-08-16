import pandas as pd

import numpy as np
from numpy import ndarray

from pandas import DataFrame

import joblib

from scipy.sparse import csr_matrix

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import set_config
from sklearn.svm import SVC

from support import *

set_config(display='text')


# PREPARE THE DATA


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



# Remove the ocean proximity column because it is not numerical (it's words, like close, far, etc.)
housing_num: DataFrame = housing.drop("ocean_proximity", axis=1)


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


# TRAIN THE MODEL


# Create predictions using a Random Forest Regressor and automatically tweak hyperparameters using a Grid Search
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features':[2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)


# Use a grid search to find the right hyperparameters
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

# Create the prediction
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)

# Print the mean test scores
cvres: dict = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

# Cross evaluate the scores
scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=10)
rmse_scores: ndarray = np.sqrt(-scores)

# Display the current scores
display_scores(rmse_scores)

# Get the best possible model from the grid search
final_model = grid_search.best_estimator_

filename = "housing_forest_reg_model.pkl"
joblib.dump(final_model, filename)

# Load up the split test data, removing the median house value from the X input data
X_test: DataFrame = test.drop("median_house_value", axis=1)
y_test: DataFrame = test["median_house_value"].copy()

# Prepare the x test data
X_test_prepared = full_pipeline.transform(X_test)

# Perform predictions
final_predictions = final_model.predict(X_test_prepared)

# Calculate RMSE from the predictions
final_mse = mean_squared_error(y_test, final_predictions)
print("RMSE", np.sqrt(final_mse))

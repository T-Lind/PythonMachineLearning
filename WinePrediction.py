import numpy as np
import pandas as pd
from pandas import DataFrame

import joblib
from pandas.plotting import scatter_matrix
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import matplotlib.pyplot as plt

from support import *

set_config(display='text')

# Get the winequality-white.csv file from my datasets
wine: DataFrame = pd.read_csv(DATASETS+"winequality-white.csv", delimiter=";")

# Find item correlations
corr_matrix: DataFrame = wine.corr()
print(corr_matrix["quality"].sort_values(ascending=False))

# Split the data
train_with_quality: DataFrame
test: DataFrame
train_with_quality, test = train_test_split(wine, test_size=0.2, random_state=42)

train: DataFrame = train_with_quality.drop("quality", axis=1)
train_labels = train_with_quality["quality"].copy()

# Find null rows in the dataset
null_rows_idx = wine.isnull().any(axis=1)
housing_option1 = wine.copy()
housing_option1.dropna(subset=["quality"], inplace=True)  # option 1

# Construct a pipeline
num_pipeline: Pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
train = num_pipeline.fit_transform(train)

attribs = list(train)


param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features':[2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

forest_reg = RandomForestRegressor()
forest_reg.fit(train, train_labels)

# Use a grid search to find the right hyperparameters
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(train, train_labels)

# Get the best model
model = grid_search.best_estimator_

# Store the best model
joblib.dump(model, "wine_forest_reg_model.pkl")

# Get the test data
X_test: DataFrame = test.drop("quality", axis=1)
y_test: DataFrame = test["quality"].copy()

# Prepare the input test data and perform predictions
X_test_prepared: DataFrame = num_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)

# Calculate RMSE from the predictions and the labels (y_test)
final_mse = mean_squared_error(y_test, final_predictions)
print("RMSE", np.sqrt(final_mse))

input_data = [[6.6, 0.3, 0.18, 7.5, 0.05, 32, 128, 0.99, 3.22, 0.50, 12]]
input_prediction = model.predict(input_data)
print(input_prediction)

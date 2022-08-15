import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame

import joblib
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from support import *

# LOAD EXTERNAL FILES

housing: DataFrame = pd.read_csv(DATASETS+"winequality-red.csv")

filename = "forest_reg_model.pkl"
model = joblib.load(filename)


set_config(display='text')


# PREPARE THE DATA

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

# Load up the split test data, removing the median house value from the X input data
X_test: DataFrame = test.drop("median_house_value", axis=1)
y_test: DataFrame = test["median_house_value"].copy()

# Prepare the x test data
X_test_prepared = full_pipeline.transform(X_test)


# Perform predictions
final_predictions: ndarray = model.predict(X_test_prepared)


# Calculate RMSE from the predictions
final_mse = mean_squared_error(y_test, final_predictions)
print("RMSE", np.sqrt(final_mse))

import statsmodels.api as sm 
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
# print(data.DESCR)

# Pandas and NumPy import
import numpy as np
import pandas as pd

# print(data.target) 

# Set the features  
df = pd.DataFrame(data.data, columns=data.feature_names)
print(df.head())

# Set the target
target = pd.DataFrame(data.target, columns=["MEDV"])

# Train a linear regression model
X = df[["AveRooms", "AveOccup"]]
y = target["MEDV"]

# Add the constant term 
# X = sm.add_constant(X) 

# Fit and make the predictions by the model
model = sm.OLS(y, X).fit()
# predictions = model.predict(X)
# print(predictions)

# Print out the statistics
print(model.summary())


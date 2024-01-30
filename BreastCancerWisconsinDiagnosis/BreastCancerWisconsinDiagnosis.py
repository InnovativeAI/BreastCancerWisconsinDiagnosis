# performing linear algebra
import numpy as np
# data processing
import pandas as pd
# visualisation
import matplotlib.pyplot as plt

data = pd.read_csv("..\\breast-cancer-wisconsin-data\\data.csv")
print(data.head)

data.info()

data.drop(['Unnamed: 32', 'id'], axis = 1)
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]

# input and output data
y = data.diagnosis.values
x_data = data.drop(['diagnosis'], axis = 1)

# normalisation
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.15, random_state=42)

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

print("x train: ", x_train.shape)
print("x test: ", x_test.shape)
print("y train: ", y_train.shape)
print("y test: ", y_test.shape)


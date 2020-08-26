# encoding: utf-8

import numpy as np
import pandas as pd


def normalize(data):
    return data.apply(lambda column: (column - column.mean()) / column.std())


# read
data = pd.read_csv("Data/data1.csv", names=["square", "bedroom", "price"])

# normalize
data_normalize = normalize(data)

# get ones
ones = pd.DataFrame({'ones': np.ones(len(data_normalize))})

# add ones
# axis : {0/'index', 1/'columns'}, default 0
ones_data_normalize = pd.concat([ones, data_normalize], axis=1)

# get X Y data
x_data = np.array(ones_data_normalize[ones_data_normalize.columns[0:3]])
y_data = np.array(ones_data_normalize[ones_data_normalize.columns[-1]]).reshape(len(ones_data_normalize), 1)

print(x_data.shape, type(x_data))
print(y_data.shape, type(y_data))
# (47, 3) <class 'numpy.ndarray'>
# (47, 1) <class 'numpy.ndarray'>

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
ones.info()

# add ones
# axis : {0/'index', 1/'columns'}, default 0
ones_data_normalize = pd.concat([ones, data_normalize], axis=1)

print("\n\n\n")
ones_data_normalize.info()
print("\n\n\n")
print(ones_data_normalize.head())
#    ones    square   bedroom     price
# 0   1.0  0.130010 -0.223675  0.475747
# 1   1.0 -0.504190 -0.223675 -0.084074
# 2   1.0  0.502476 -0.223675  0.228626
# 3   1.0 -0.735723 -1.537767 -0.867025
# 4   1.0  1.257476  1.090417  1.595389

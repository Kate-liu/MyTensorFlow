# encoding: utf-8

import pandas as pd
import matplotlib.pyplot as plt


def normalize(data):
    return data.apply(lambda column: (column - column.mean()) / column.std())


data = pd.read_csv("Data/data1.csv", names=["square", "bedroom", "price"])
print(data.head())
#    square  bedroom   price
# 0    2104        3  399900
# 1    1600        3  329900
# 2    2400        3  369000
# 3    1416        2  232000
# 4    3000        4  539900

data_normalize = normalize(data)
print("\n\n\n")
print(data_normalize.head())
#      square   bedroom     price
# 0  0.130010 -0.223675  0.475747
# 1 -0.504190 -0.223675 -0.084074
# 2  0.502476 -0.223675  0.228626
# 3 -0.735723 -1.537767 -0.867025
# 4  1.257476  1.090417  1.595389


fig = plt.figure()
ax = plt.axes(projection="3d")
ax.set_xlabel("square")
ax.set_ylabel("bedrooms")
ax.set_zlabel("price")
ax.scatter3D(data_normalize["square"], data_normalize["bedroom"], data_normalize["price"], c=data_normalize["price"], cmap="Reds")
plt.show()

# show detail
print("\n\n\n")
data.info()
print("\n\n\n")
data_normalize.info()

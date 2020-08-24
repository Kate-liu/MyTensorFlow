# encoding: utf-8

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Data/data1.csv", names=["square", "bedroom", "price"])
print(data.head())
#    square  bedroom   price
# 0    2104        3  399900
# 1    1600        3  329900
# 2    2400        3  369000
# 3    1416        2  232000
# 4    3000        4  539900


fig = plt.figure()

ax = plt.axes(projection="3d")
ax.set_xlabel("square")
ax.set_ylabel("bedrooms")
ax.set_zlabel("price")
ax.scatter3D(data["square"], data["bedroom"], data["price"], c=data["price"], cmap="Greens")

plt.show()

# encoding: utf-8

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# def set(context="notebook", style="darkgrid", palette="deep",
#         font="sans-serif", font_scale=1, color_codes=True, rc=None):
sns.set(context="notebook", style="whitegrid", palette="dark")

# get data
data = pd.read_csv("Data/data0.csv", names=["square", "price"])

# show data
sns.lmplot("square", "price", data, height=8, fit_reg=False)
plt.show()

# show data and regression plot
sns.lmplot("square", "price", data, height=8, fit_reg=True)
plt.show()

# show detail
data.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 47 entries, 0 to 46
# Data columns (total 2 columns):
# square    47 non-null int64
# price     47 non-null int64
# dtypes: int64(2)
# memory usage: 880.0 bytes
# None

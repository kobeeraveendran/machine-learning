import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# white wine data
white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep = ";")

# red wine data
red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep = ";")

# take a look at how the data is formatted in the dataframe (optional)
# note that the output is similar to R's summary() method
print(white.info())
print(red.info())

# DATA VISUALIZATION

## ALCOHOL CONTENT
fig, ax = plt.subplots(1, 2)

ax[0].hist(red.alcohol, 10, facecolor = "red", ec = "black", alpha = 0.5, label = "red wine")
ax[1].hist(white.alcohol, 10, facecolor = "white", ec = "black", lw = 0.5, alpha = 0.5, label = "white wine")

# standardize axes
ax[0].set_ylim([0, 1000])
ax[0].set_xlabel("alcohol in % vol")
ax[0].set_ylabel("Frequency")

ax[1].set_xlabel("alcohol in % vol")
ax[1].set_ylabel("Frequency")

fig.suptitle("Distribution of alcohol in % vol")

plt.show()

# NOTE: you can also get a feel for distributions in the data by using
# np.histogram(), like so:
print(np.histogram(red.alcohol, bins = [7, 8, 9, 10, 11, 12, 13, 14, 15]))
print(np.histogram(white.alcohol, bins = [7, 8, 9, 10, 11, 12, 13, 14, 15]))

# the output shows that the highest should be around 9, which aligns with what the graph says

## SULPHATES
fig, ax = plt.subplots(1, 2, figsize = (8, 4))

ax[0].scatter(red['quality'], red['sulphates'], color = "red")
ax[1].scatter(white['quality'], white['sulphates'], color = "white", edgecolors = "black", lw = 0.5)

ax[0].set_title("red wine")
ax[1].set_title("white wine")
ax[0].set_xlabel("quality")
ax[1].set_xlabel("quality")
ax[0].set_ylabel("sulphates")
ax[1].set_ylabel("sulphates")
ax[0].set_xlim([0, 10])
ax[1].set_xlim([0, 10])
ax[0].set_ylim([0,, 2.5])
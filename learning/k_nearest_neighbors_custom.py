# Kobee Raveendran
# k-nearest neighbors algorithm from scratch
# March 6, 2018

import numpy as np 
import matplotlib.pyplot as plt 
import warnings
from matplotlib import style
from math import sqrt
from collections import Counter

style.use('fivethirtyeight')

# example of calculating euclidean distance in a clear
'''
plot1 = [1,3]
plot2 = [2,5]
euclidean_distance = sqrt((plot1[0] - plot2[0]) ** 2 + (plot1[1] - plot2[1]) ** 2)
print(euclidean_distance)
'''

# sample dataset with two classes (features k and r)
dataset = {'k': [[1,2], [2,3], [3,1]], 'r':[[6,5], [7,7], [8,6]]}

new_features = [5,7]



# k-nearest neighbors function, with default k-value of 3
def k_nearest_neighbors(data, predict, k = 3):
    if len(data) >= k:
        warnings.warn("K is set to a value less than the number of classes; change and try again.")

    distances = []
    for group in data:
        for features in data[group]:
            # could use the straightforward euclidean distance alg from above, but numpy's norm is faster,
            # and does not apply to dimensions other than  d = 2
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    print(Counter(votes).most_common)
    vote_result = Counter(votes).most_common(1)[0][0]
     
    return vote_result

result = k_nearest_neighbors(dataset, new_features, k = 3)
print(result)

for i in dataset:
    for ii in dataset[i]:
        plt.scatter(ii[0], ii[1], s = 100, color = i)

plt.scatter(new_features[0], new_features[1], color = result, s = 100)
plt.show()

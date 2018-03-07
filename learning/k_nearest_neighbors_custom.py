# Kobee Raveendran
# k-nearest neighbors algorithm from scratch
# March 6, 2018

import numpy as np 
import matplotlib.pyplot as plt 
import warnings
from matplotlib import style
from math import sqrt
from collections import Counter
import pandas as pd 
import random

style.use('fivethirtyeight')

# example of calculating euclidean distance in a clear
'''
plot1 = [1,3]
plot2 = [2,5]
euclidean_distance = sqrt((plot1[0] - plot2[0]) ** 2 + (plot1[1] - plot2[1]) ** 2)
print(euclidean_distance)


# sample dataset with two classes (features k and r)
dataset = {'k': [[1,2], [2,3], [3,1]], 'r':[[6,5], [7,7], [8,6]]}

new_features = [5,7]
'''


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
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k
     
    return vote_result, confidence

accuracies = []

for i in range(25):
    df = pd.read_csv('breast-cancer-wisconsin.data.txt')
    df.replace('?', -99999, inplace = True)
    df.drop(['id'], 1, inplace = True)

    full_data = df.astype(float).values.tolist()

    random.shuffle(full_data)

    test_size = 0.4
    train_set = {2:[], 4:[]}
    test_set = {2:[], 4:[]}
    train_data = full_data[:-int(test_size * len(full_data))]
    test_data = full_data[-int(test_size * len(full_data)):]

    for i in train_data:
        train_set[i[-1]].append(i[:-1])

    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    correct = 0
    total = 0

    for group in test_set:
        for data in test_set[group]:
            vote, confidence = k_nearest_neighbors(train_set, data, k = 5)
            if group == vote:
                correct += 1
            total += 1

    #print("correct:", correct)
    #print("total", total)
    #print("Accuracy:", correct / total)
    accuracies.append(correct / total)

print(sum(accuracies) / len(accuracies))

# not included in testing phase
'''
result = k_nearest_neighbors(dataset, new_features, k = 3)
print(result)

for i in dataset:
    for ii in dataset[i]:
        plt.scatter(ii[0], ii[1], s = 100, color = i)

plt.scatter(new_features[0], new_features[1], color = result, s = 100)
plt.show()
'''
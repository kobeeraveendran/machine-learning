import numpy as np 
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn import neighbors
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace = True)
df.drop(['id'], 1, inplace = True)

# takes everything except the 'class' column (features)
X = np.array(df.drop(['class'], 1))
# makes y the 'class' column (labels)
y = np.array(df['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)
clf = neighbors.KNeighborsClassifier()

# train classifier
clf.fit(X_train, y_train)

# test classifier
accuracy = clf.score(X_test, y_test)
print("Model accuracy:", accuracy * 100, "%")

# predicting on new data
example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 1, 1, 1, 2, 3, 2, 1]])
example_measures = example_measures.reshape(len(example_measures),-1)
prediction = clf.predict(example_measures)
print('Prediction:', prediction)
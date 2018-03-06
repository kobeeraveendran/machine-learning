import pandas as pd
import quandl
import math
import numpy as np 
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import datetime
import matplotlib.pyplot as plt
import pickle
from matplotlib import style

style.use('ggplot')

# API Key for quandl datasets
quandl.ApiConfig.api_key = '4vLqk_aXmFCTvhfk7pA_'

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume',]]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
# instead of getting rid of entire NA columns, replaces with -99999
df.fillna(-99999, inplace = True)

# prints out 10% of the dataframe
forecast_out = int(math.ceil(0.01 * len(df)))
#print("Number of days into the future:", forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
df.dropna(inplace = True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)
# now that the classifier has been pickled (saved), retraining isn't necessary
'''
# create classifier and fit training data
clf = LinearRegression(n_jobs = -1)  #n_jobs changes how many jobs to run at at time; -1 goes as fast as processor allows
clf.fit(X_train, y_train)

with open('regression.pickle', 'wb') as f:
    pickle.dump(clf, f)
'''
pickle_in = open('regression.pickle', 'rb')

clf = pickle.load(pickle_in)
# tests accuracy of classifier
accuracy = clf.score(X_test, y_test)

#print("Model accuracy:", accuracy)

forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
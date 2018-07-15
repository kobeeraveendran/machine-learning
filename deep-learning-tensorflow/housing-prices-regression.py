import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

housing = keras.datasets.boston_housing

(train_data, train_labels), (test_data, test_labels) = housing.load_data()

print('Training set shape: {} \nTesting set shape: {} \n'.format(train_data.shape, test_data.shape))
print(train_data[0])

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 
                'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 
                'B', 'LSTAT']

df = pd.DataFrame(train_data, columns = column_names)
print(df.head())
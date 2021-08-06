# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 00:12:50 2020

@author: Marin
"""
#Apple Stock Prediction
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_train = pd.read_csv("AAPl_train.csv")
training_set = dataset_train.iloc[:,1:2].values

#Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

#Creating
x_train = []
y_train = []
for i in range(60, 1687):
    x_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
x_train = np.array(x_train)
y_train = np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#RNN
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential

regressor = Sequential()

regressor.add(LSTM(units = 60, input_shape=(x_train.shape[1], 1), return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 60, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 60, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = False))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(x_train,y_train, epochs = 100, batch_size = 32)

dataset_test = pd.read_csv('AAPL_test.csv')
real_prices = dataset_test.iloc[:,1:2].values

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

x_test = []
for i in range (60, len(inputs)):
    x_test.append(inputs[i-60:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = regressor.predict(x_test)
predicted_prices = sc.inverse_transform(predicted_prices)

#visualising
plt.plot(real_prices, color = 'red', label = 'Prava cijena')
plt.plot(predicted_prices, color = 'blue', label = 'Predvidena cijena')
plt.title("Apple pretpostavka cijene")
plt.xlabel('Vrijeme')
plt.ylabel('Cijena')
plt.legend()
plt.show()








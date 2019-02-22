#!/usr/bin/python
#coding:utf8

'''
first step:pip install pyramid-arima
'''

import pandas as pd
import numpy as np
from pyramid.arima import auto_arima

print("load dataset...")

data = pd.read_csv('international-airline-passengers.csv')
#divide into train and validation set
train = data[:int(0.7*(len(data)))]
valid = data[int(0.7*(len(data))):]

#preprocessing (since arima takes univariate series as input)
train.drop('Month',axis=1,inplace=True)
valid.drop('Month',axis=1,inplace=True)

#plotting the data
#train['International airline passengers'].plot()
#valid['International airline passengers'].plot()

print("train the model")
model = auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True)
model.fit(train)

forecast = model.predict(n_periods=len(valid))
forecast = pd.DataFrame(forecast,index = valid.index,columns=['Prediction'])

#plot the predictions for validation set
#plt.plot(train, label='Train')
#plt.plot(valid, label='Valid')
#plt.plot(forecast, label='Prediction')
#plt.show()

print("evalute the prediction...")
#calculate rmse
from math import sqrt
from sklearn.metrics import mean_squared_error

rms = sqrt(mean_squared_error(valid,forecast))
print(rms)

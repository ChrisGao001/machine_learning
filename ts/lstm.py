import prophet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10

data = pd.read_csv("./data.csv", sep='\t')
data.drop('Unnamed: 4',axis=1, inplace=True )

# 准备样本
df = data.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['date', 'pv'])

for i in range(0,len(df)):
    new_data['date'][i] = df['date'][i]
    new_data['pv'][i] = df['pv'][i]
    
new_data.index = new_data.date
new_data.drop('date', axis=1, inplace=True)
print(new_data.head())

train = new_data[:93].values
valid = new_data[93:].values

def lstm_model_train(x_train, y_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=10, batch_size=1, verbose=2)
    return model

dataset = new_data.values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
x_train, y_train = [], []

for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
    
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
model = lstm_model_train(x_train, y_train)

inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)
X_test = []

for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)
print(X_test.shape)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
preds = model.predict(X_test)
preds = scaler.inverse_transform(preds)
error = rmse(preds, valid)
print("rmse:{0}".format(error))
train = new_data[:93]
valid = new_data[93:]
valid['pred'] = preds

plt.plot(train['pv'])
plt.plot(valid[['pv','pred']])



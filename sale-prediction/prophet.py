'''
installation:
pip install update numpy==1.14.5
pip install update fbprophet==0.3.post2 
https://github.com/facebook/prophet/issues/808
'''

import pandas as pd
import numpy as np
from fbprophet import Prophet

def rmse(preds, y):
    result = np.sqrt(np.mean(np.power(preds - y, 2)))
    return result

# load data
data = pd.read_csv("./data.csv", sep='\t')
new_data = pd.DataFrame(index=range(0,len(data)),columns=['date', 'pv'])
 
for i in range(0,len(data)):
    new_data['date'][i] = data['date'][i]
    new_data['pv'][i] = data['pv'][i]

new_data["date"] = pd.to_datetime(new_data["date"], format="%Y-%m-%d")
new_data.index = new_data["date"]
new_data.rename(columns={'pv': 'y', 'date': 'ds'}, inplace=True)

train = new_data[:93]
valid = new_data[93:]["y"].values
# create model
model = Prophet()
model.fit(train)

# predict
print("len(valid)={0}".format(len(valid)))
future = model.make_future_dataframe(periods=len(valid))
print(future.head(100))
forecast = model.predict(future)
print(forecast.head())
print(forecast.shape)
preds = forecast['yhat'][93:].values
error = rmse(preds, valid)
print("rmse:{0}".format(error))




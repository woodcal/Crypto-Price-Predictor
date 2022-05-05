from logging.handlers import TimedRotatingFileHandler
from optparse import Values
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import pandas_datareader as web
import datetime as dt 

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layer import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

cryptoCurrency = "ETH" #Cryptocurrency to Predict
againstCurrency = "USD" #Price to compare it to

start = dt.datetime(2016, 1, 1)
end = dt.datetime.now()

data = web.DataReader(f'{cryptoCurrency}-{againstCurrency}', 'yahoo', start, end)

#Prepare the Data for the Neural Network
#print(data.head()) shows closing values of bitcoin each day
scaler = MinMaxScaler(feature_range=(0, 1))
scaledData = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

predictionDays = 60 #Days used as test data
futureDays = 0 #Days ahead of test data to be predicted

xTrain, yTrain = [], []

for x in range(predictionDays, len(scaledData)-futureDays):
    xTrain.append(scaledData[x-predictionDays:x, 0])
    yTrain.append(scaledData[x+futureDays, 0])

xTrain, yTrain = np.array(xTrain), np.array(yTrain)
xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))

#Create the Neural Network

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(xTrain.shape[1], 1)))
model.add(Droupout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Droupout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2)
model.add(Dense(units=1))

model.compile(optimizer='adam', loss ='mean_squared_error')
model.fit(xTrain, yTrain, epochs=25, batch_size=32)

#Testing the Model

testStart = dt.datetime(2020, 1, 1)
testEnd = dt.datetime.now()

testData = web.DataReader(f'{cryptoCurrency}-{againstCurrency}', 'yahoo', testStart, testEnd)
actualPrices = test_data['Close'].values
totalDataSet = pd.concat((data['Close'],testData['Close']), axis=0)

modelInputs = totalDataSet[len(totalDataSet) - len(testData) - predictionDays: ].value
modelInputs = modelInputs.reshape(-1, 1)
modelInputs = scaler.fit_transform(modelInputs)

x_test = []

for x in range(predictionDays, len(modelInputs)):
    x_test.append(modelInputs[x - predictionDays: x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictioinPrices = model.predict(x_test)
predictionPrices = scaler.inverse_transform(predictionPrices)

plt.plot(actualPrices, color='black', label='Actual Prices')
plt.plot(predictionPrices, color='green', label='Prediction Prices')
plt.title(cryptoCurrency +  'Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc='upper_left')
plt.show()

#Predict next day

realData = [modelInputs[len(modelInputs) + 1 - predictionDays: len(modelInputs) + 1, 0]]
realData = np.array(realData)
realData = np.reshape(realData, (realData.shape[0], realData.shape[1], 1))

prediction = model.predict(realData)
prediction = scaler.inverse_transform(prediction)
print(prediction)
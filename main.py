# Imports

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Load Data

company = 'AAPL'  # -> ticker symbol
epochs = 100
batchsize = 10

start = dt.datetime(2010, 1, 1)
end = dt.datetime(2020, 1, 1)

data = web.DataReader(company, 'yahoo', start, end)

# Prepare Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))  # Closing price is predicted

prediciton_days = 70  # For prediciton: how many days look back? Here 70 days

x_train = []
y_train = []

for x in range(prediciton_days, len(scaled_data)):
    x_train.append(scaled_data[x - prediciton_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build model

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))  # Numbers to experiment
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))  # Numbers to experiment
model.add(Dropout(0.2))
model.add(LSTM(units=50))  # Numbers to experiment
model.add(Dropout(0.2))
model.add(Dense(units=1))  # Prediciton of the next closing price

model.compile(optimizer='adam', loss='mean_squared_error')  # different optimizer functions!
model.fit(x_train, y_train, epochs=epochs,
          batch_size=batchsize)  # Batchsize: Model sees xx batches at once all the time

# Test the Model Accuracy on Existing Data
# Load Test Data

test_start = dt.datetime.now() - dt.timedelta(150)
test_end = dt.datetime.now()

test_data = web.DataReader(company, 'yahoo', test_start, test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediciton_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

# Make predictions on test data

x_test = []

for x in range(prediciton_days, len(model_inputs)):
    x_test.append(model_inputs[x - prediciton_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plot test predictions

plt.plot(actual_prices[-150:], color="black", label="actual_price")
plt.plot(predicted_prices[-150:], color="green", label="predicted_price")
plt.title(f"{company} Share Price - Actual vs Predicted last 150 days")
plt.xlabel('Time in Days')
plt.ylabel('Price')
plt.legend()
plt.show()
plt.savefig(f"stock_img/{company, epochs, batchsize}.png")

# Predict Next Day
real_data = [model_inputs[len(model_inputs) + 1 - prediciton_days:len(model_inputs + 1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

pred = model.predict(real_data)
pred = scaler.inverse_transform(pred)
print("Prediciton:", pred)

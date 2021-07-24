import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler

data_path = '/home/vargha/Desktop/Codes/Deep Learning'
data = pd.read_csv(os.path.join(data_path, 'all_stocks_5yr.csv'))
cl = data[data['Name'] == 'MMM'].Close

# preprocessing
scale = MinMaxScaler()
cl = cl.values.reshape(cl.shape[0], 1)
cl = scale.fit_transform(cl)


def preprocessing(data, sequences_length):
    x, y = [], []
    for i in range(len(data) - sequences_length - 1):
        x.append(data[i: (i + sequences_length), 0])
        y.append(data[(i + sequences_length), 0])
    return np.array(x), np.array(y)


X, Y = preprocessing(cl, 7)
x_train, x_test = X[:int(X.shape[0] * 0.80)], X[int(X.shape[0] * 0.80):]
y_train, y_test = Y[:int(Y.shape[0] * 0.80)], Y[int(Y.shape[0] * 0.80):]
x_train = np.expand_dims(x_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)

# Model
inputs = Input((7, 1))
x = LSTM(256)(inputs)
outputs = Dense(1)(x)
model = Model(inputs, outputs)
model.summary()
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=300, shuffle=False)

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()

x_predicted = model.predict(x_test)
plt.plot(scale.inverse_transform(y_test.reshape(-1, 1)), label='true')
plt.plot(scale.inverse_transform(x_predicted), label='pred')
plt.legend()
plt.show()

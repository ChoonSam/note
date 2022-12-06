import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
import pandas as pd

data = pd.read_csv("C:/Users/82102/OneDrive/바탕 화면/최종과 최초/공사일보(z-0001)CSV.csv")
x = data.iloc[:,[1,3,4,5,6,7,8]],
y = data.iloc[:,[9]]

model = Sequential()
model.add(Dense(1, input_dim=7, activation='sigmoid'))
sgd = optimizers.SGD(learning_rate=0.001)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['binary_accuracy'])

model.fit(x, y, epochs=200)
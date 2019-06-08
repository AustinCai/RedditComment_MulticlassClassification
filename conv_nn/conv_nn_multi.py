import pickle
import re
import os
import numpy
from numpy import array
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras import layers
import keras
import matplotlib.pyplot as plt
import helper

import constants
import word_embeddings

X = helper.undump("X+20.pkl")
y = helper.undump("y+20.pkl")


print(X)
print(y)
print(len(X))
print(len(y))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1000)


input_dim = X_train.shape[1] # number of features

model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu', use_bias=True))
model.add(layers.Dense(20, activation='softmax', use_bias=True))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


history = model.fit(X_train, y_train, epochs=50, verbose=False, validation_data=(X_test, y_test), batch_size=10)

model.summary()

train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(train_accuracy))

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy: {:.4f}".format(test_accuracy))

helper.dump("history.pkl", history)



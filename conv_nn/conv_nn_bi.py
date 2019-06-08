import pickle
import re
import os
import numpy
from numpy import array
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras import layers
import matplotlib.pyplot as plt
import helper

import constants
import word_embeddings

X = helper.undump("data/X_full.pkl")
y = helper.undump("data/y_full.pkl")

# temporary; classifies everything as either in AskReddit or in NBA
for i in range(len(y)):
	if y[i] > 0:
		y[i] = 1

# balances the dataset 
X = numpy.append(X[:41045], X[41045::19], 0) #41045 is where the o switches to 1 (y[41045] is the first 1)
y = numpy.append(y[:41045], y[41045::19])

print(X)
print(y)
print(len(X))
print(len(y))



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1000)


input_dim = X_train.shape[1] # number of features

model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


history = model.fit(X_train, y_train, epochs=20, verbose=False, validation_data=(X_test, y_test), batch_size=10)

model.summary()

train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(train_accuracy))

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy: {:.4f}".format(test_accuracy))

helper.dump("history.pkl", history)



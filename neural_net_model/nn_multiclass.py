# machine learning imports
import numpy
from numpy import array
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras import layers
import keras

# utility imports
import pickle

# helper file imports
import constants
import helper

# ====================================================================================

X = helper.undump("../../data/X_mid+20.pkl")
y = helper.undump("../../data/y_mid+20.pkl")
print("dataset loaded")

print(X)
print(y)
print(len(X))
print(len(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1000)
print("data set split")

input_dim = X_train.shape[1] # number of features

model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu', use_bias=True))
model.add(layers.Dense(20, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("model built")

history = model.fit(X_train, y_train, epochs=50, verbose=2, validation_data=(X_test, y_test), batch_size=10)
print("model trained")

model.summary()

train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(train_accuracy))

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy: {:.4f}".format(test_accuracy))

helper.dump("history.pkl", history)



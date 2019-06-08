import pickle
import re
import os
import numpy
from numpy import array
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras import layers
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# function sourced from: https://realpython.com/python-keras-text-classification/
def plot_history(history):
    print("history called")
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show

def dump(dumpfile, data):
    with open(dumpfile, 'wb') as f:
        pickle.dump(data, f)

def undump(dumpfile):
    with open(dumpfile, 'rb') as f:
        data = pickle.load(f, encoding='latin1') 
    return data


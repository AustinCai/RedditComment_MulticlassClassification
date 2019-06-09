# machine learning imports
import numpy
from numpy import array
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras import layers
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# utility imports 
import pickle
import re

# helper file imports 
import constants

# Naive bayes imports
import sys
sys.path.insert(0, '../naive_bayes_baseline')
import nb_constants
import nb_helper

# ====================================================================================

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

# takes in a dataset entry (list containing author, date, id, comment string, etc)
# returns a 320 dimension feature vector (300 features form GloVe + 20 features from Naive Bayes)
def getCommentVec(datasetEntry, wordToVec, wordProbByClass):

    # extracts list of words in the comment from the dataset entry
    comment = datasetEntry[constants.COMMENT_INDEX].split() 
    commentVec = [0.0 for _ in range(constants.WORDVEC_LEN)]

    for i in range(len(comment)): # iterates through each word of the comment
        normalizedWord = re.sub(r'[^\w\s]','',comment[i].lower())

        # skip to the first word with an embedding
        while i < len(comment) and wordToVec[normalizedWord] == 0: 
            i+=1

        # if the dict doesn't contain any word mappings, don't try to query for one
        if i == len(comment): continue

        # get the word feature vector and add it to the comment feature vector
        wordVec = wordToVec[normalizedWord]
        commentVec = [commentVec[i] + wordVec[i] for i in range(constants.WORDVEC_LEN)]

    # append 20 features
    weightByClass = nb_helper.classWeightsFromComment(datasetEntry, wordProbByClass)

    commentVec += weightByClass

    return commentVec

def dump(dumpfile, data):
    with open(dumpfile, 'wb') as f:
        pickle.dump(data, f)

def undump(dumpfile):
    with open(dumpfile, 'rb') as f:
        data = pickle.load(f, encoding='latin1') 
    return data

def progressUpdate(iterator, target, updateCount):
    if iterator % (target/updateCount) == 0:
        print("Progress: {}% ({}/{}) complete".format(round(100*(iterator/target), 1), iterator, target))





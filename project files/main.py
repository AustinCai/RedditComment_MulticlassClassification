import importlib
import sys
import pickle
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from random import shuffle
pnn = importlib.import_module("pytorch-neural-network")
we = importlib.import_module("word_embeddings")

""" Global Parameters """
#TODO: put all this stuff in a library file
# Load some data
wordEmbedDict = we.getWordEmbeddingDict() # Load the dictionary
keyset = wordEmbedDict.keys()

# Indices of our desired data
TRUE_LABEL = 8 # Index of the true label, hard coded
BODY_INDEX = 17 # Hard coded because of the structure of our depickled data

# Neural Network Parameters
NUM_FEATURES = 300
NUM_EXAMPLES = 3
NUM_ITERATIONS = 1000
SUBREDDIT = "leagueoflegends"
unparsed = "../data/condensed_dataset_SMALL.pkl" # will change this so we can just call the pkl set
""" """

# @param dir: string, directory of pickle data
# @return dataset: unpickled dataset
def loadPickleData(dir):
	with open(dir, 'rb') as f:
		dataset = pickle.load(f)
	return dataset

# Returns X with m examples, Y
def loadData(pickleDir, m):
	pickle = loadPickleData(pickleDir)
	return vectorizeDataSet(pickle, m)

def vectorizeWord(word):
	vWord = torch.zeros((NUM_FEATURES, 1), dtype=torch.float)
	if word in keyset:
		vWord = torch.reshape(torch.FloatTensor(wordEmbedDict[word]), (NUM_FEATURES, 1))
	return vWord

def vectorizeComment(body):
	vComment = torch.zeros((NUM_FEATURES, 1), dtype=torch.float)
	words = body.split()
	for word in words:
		vWord = vectorizeWord(word)
		vComment += vWord
	return vComment

# Given label as string, return tensor 1 if SUBREDDIT, tensor 0 if not SUBREDDIT
def parseLabel(labelStr):
	if labelStr == SUBREDDIT:
		return torch.tensor(([1]), dtype=torch.float)
	return torch.tensor(([0]), dtype=torch.float)

def vectorizeDataSet(data, m):
	shuffle(data)
	testSet = data.pop()
	unrollComment = data[0][BODY_INDEX]
	X = vectorizeComment(unrollComment)
	unrollLabel = data[0][TRUE_LABEL]
	Y = parseLabel(unrollLabel)

	# For each example in old data set, get the actual comment and featurize it into X
	# Also get its true label
	for i in range(1, m):
		comment = data[i][BODY_INDEX]
		example = vectorizeComment(comment)
		label = parseLabel(data[i][TRUE_LABEL])

		Y = torch.cat((Y, label))
		X = torch.cat((X, example), 1)
	return [X, Y, testSet]

def main():
	# Goal: single hidden layer neural network with 3 neurons in it
	# TODO: Paramterize neural network
	# numNeurons
	# numHiddenLayers

	NN = pnn.Neural_Network()

	# In this iteration X is our training set and xPredicted is our test set
	# Will implement validation set soon
	# TODO: Validation set

	X, Y, testSet = loadData(unparsed, NUM_EXAMPLES)
	# TODO: Change these to a data set X where each column is a feature with m training examples
	# Dummy data for now to implement 
	# Create data set and labels
	# X = torch.tensor(([4, 4, 5, 2, 1], [7, 9, 8, 5, 4]), dtype=torch.float) # Dummy data
	# y = torch.tensor(([82, 100, 85, 60, 30]), dtype=torch.float) # True label test
	# xPredicted = torch.tensor(([4], [7]), dtype=torch.float) 

	# scale units
	# TODO: Scale by subtracting the mean and divide by the standard deviation
	# X_max, _ = torch.max(X, 0)
	# xPredicted_max, _ = torch.max(xPredicted, 0)
	# X = torch.div(X, X_max)
	# xPredicted = torch.div(xPredicted, xPredicted_max)
	# y = y / 100  # max test score is 100

	# Test our model:
	testComment = testSet[BODY_INDEX]
	testTrueLabel = testSet[TRUE_LABEL]
	testX = vectorizeComment(testComment)
	testY = parseLabel(testTrueLabel)



	for i in range(NUM_ITERATIONS): 
		# This print statement outputs the value of our loss function. If it is decreasing, then we are doing well
	    # print("#" + str(i) + " Loss: " + str(torch.mean((y - NN(X))**2).detach().item()))  # mean sum squared loss
	    NN.train(X, Y)

	# NN.saveWeights(NN), learn how to use this
	NN.predict(testX)
	print("Label we are testing: ", SUBREDDIT)
	print("Label with tensor value: {}, {}".format(testTrueLabel, testY))

main()

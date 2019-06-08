# machine learning imports
from numpy import array

# utility imports
import pickle
import re

# helper file imports
import constants
import helper

# Naive bayes imports
import sys
sys.path.insert(0, '../naive_bayes')
import nb_constants
import nb_helper

# ====================================================================================

# load dataset, a list of entires where each entry is a list containing author, date, id, comment string, etc.
dataset = helper.undump(constants.FULL_DATASET)
print("dataset loaded")

# load word embedding dictionaries 
wordToVec = helper.undump("../../data/embeddings_100k.pkl")
print("wordToVec loaded")

# learn probabilities to calculate Naive Bayes features
wordProbByClass = nb_helper.learnProbsMulti(dataset)
print("wordProbByClass calculated")

X = []
y = []

for k in range(len(dataset)): # iterates through each comment
	
	helper.progressUpdate(k, len(dataset), 1000)

	# returns a 320 dimension feature vector (300 features form GloVe + 20 features from Naive Bayes)
	commentVec = helper.getCommentVec(dataset[k], wordToVec, wordProbByClass)

	X.append(commentVec)
	y.append(dataset[k][-1])
	
print("X and y created")

print(X[0])
print(len(X[0]))

X = array(X)
y = array(y)

print(X)

helper.dump('X_full+20.pkl', X)
helper.dump('y_full+20.pkl', y)



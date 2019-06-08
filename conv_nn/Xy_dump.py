import pickle
import re
import os
from numpy import array

import sys
sys.path.insert(0, '../naive_bayes')

import constants
import word_embeddings
import nb_constants
import nb_helper

# # X is a list (numpy.ndarray) of 150 entries, where each entry is a R^4 feature vector
# # y is a list (numpy.ndarray) of 150 classification labels, where y[i] is the label for X[i]
# X, y = load_iris(return_X_y=True)

# print(X)
# print(y)


with open(constants.FULL_DATASET, 'rb') as f:
    dataset = pickle.load(f) 

print("dataset loaded")
comments = [entry[constants.COMMENT_INDEX].split() for entry in dataset]
#---------------------------------------------

# NB features
wordProbByClass = nb_helper.learnProbsMulti(dataset)

print("wordProbByClass calculated")

# word to vec features
with open("../../data/embeddings.pkl", 'rb') as f:
    wordToVec = pickle.load(f) 

print("wordToVec loaded")

X = []
y = []
for k in range(len(comments)): # iterates through each comment
	comment = comments[k]

	commentVecAdditions = 0
	commentVec = [0.0 for _ in range(constants.WORDVEC_LEN)]

	for i in range(len(comment)): # iterates through each word of the comment

		# skip to the first word with an embedding
		while i < len(comment) and wordToVec[re.sub(r'[^\w\s]','',comment[i].lower())] == 0: 
			i+=1

		# if the dict doesn't contain any word mappings, don't try to query for one
		if i == len(comment): continue

		wordVec = wordToVec[re.sub(r'[^\w\s]','',comment[i].lower())]
		commentVec = [commentVec[i] + wordVec[i] for i in range(constants.WORDVEC_LEN)]
		commentVecAdditions += 1

	# if mappings exist, average them out
	if i != len(comment): 
		commentVec = [commentVec[i]/commentVecAdditions for i in range(len(commentVec))]

	# append 20 features
	weightByClass = nb_helper.classWeightsFromComment(dataset[k], wordProbByClass)

	# TODO: normalize?

	commentVec += weightByClass

	X.append(commentVec)
	y.append(dataset[k][-1])
	
print("X and y created")

print(X[0])
print(len(X[0]))

X = array(X)
y = array(y)

print(X)

with open('X_full+20.pkl', 'wb') as f:
    pickle.dump(X, f)
with open('y_full+20.pkl', 'wb') as f:
    pickle.dump(y, f)
import pickle
import re
import os
import numpy
from numpy import array
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

import constants
import word_embeddings

with open("X.pkl", 'rb') as f:
    X = pickle.load(f) 
with open("y.pkl", 'rb') as f:
    y = pickle.load(f) 

clf = LogisticRegression(random_state = 0, solver='lbfgs', multi_class='multinomial', max_iter = 10000000).fit(X, y)

for i in range(constants.CATEGORY_COUNT):
	predict = clf.predict(X[(constants.TRAINSET_CAT_LEN)*i:(constants.TRAINSET_CAT_LEN)*(i+1), :])
	#predictProb = clf.predict_proba(X[:2, :])

	unique, counts = numpy.unique(predict, return_counts = True)
	predictDict = dict(zip(unique, counts))

	
	# print(predict)
	# print(y[400*i:400*(i+1)])
	# print(predictDict)

	print(predictDict[i]/(constants.TRAINSET_LEN/constants.CATEGORY_COUNT))
	

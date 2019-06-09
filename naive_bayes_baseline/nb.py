import pickle
import collections
import nb_helper
import re 
import nb_constants

def run_nb(dataset = None):

	top_20 = ['AskReddit', 'leagueoflegends', 'nba', 'funny', 'pics', 'nfl', 'pcmasterrace', \
	    'videos', 'news', 'todayilearned', 'DestinyTheGame', 'worldnews', 'soccer', 'DotA2', \
	    'AdviceAnimals', 'WTF', 'GlobalOffensive', 'hockey', 'movies', 'SquaredCircle']

	if dataset == None:
		with open(nb_constants.FULL_DATASET, 'rb') as f:
		    dataset = pickle.load(f) 

	trainSet, testSet = nb_helper.partitionDataset(dataset)

	wordProbByClass = nb_helper.learnProbsMulti(trainSet)

	countByClass, correctCountByClass, weightByClassList = nb_helper.classAndEvalMulti(testSet, wordProbByClass)

	accuracyByClass = [float(correctCountByClass[i])/10000 for i in range(nb_constants.CATEGORY_COUNT)]

	print accuracyByClass

	results = collections.defaultdict(int)
	for i in range(len(top_20)):
		results[top_20[i]] = accuracyByClass[i]

	return results, weightByClassList

results, weightByClassList = run_nb()
print results
import pickle
import collections
import string
import re
import nb_constants

def progressUpdate(iterator, target, updateCount):
    if iterator % (target/updateCount) == 0:
        print("Progress: {}% ({}/{}) complete".format(round(100*(iterator/target), 1), iterator, target))


def partitionDataset(dataset): 
	trainSet = []
	testSet = []

	for i in range(len(dataset)):
		if i % (len(dataset)/nb_constants.CATEGORY_COUNT) >= (len(dataset)/nb_constants.CATEGORY_COUNT) * nb_constants.TESTSET_PROPORTION:
			testSet.append(dataset[i])
		else:
			trainSet.append(dataset[i])

	return trainSet, testSet


def learnProbsMulti(trainSet):

	wordCountByClass = []
	for _ in range(nb_constants.CATEGORY_COUNT): wordCountByClass.append(collections.defaultdict(float))

	for i in range(len(trainSet)):
		progressUpdate(i, len(trainSet), 100)

		entry = trainSet[i]

		entryList = entry[nb_constants.COMMENT_INDEX].split()
		for word in entryList:
			normalizedWord = re.sub(r'[^\w\s]','',word.lower())
			wordCountByClass[entry[-1]][normalizedWord] += 1

	wordProbByClass = []
	for _ in range(nb_constants.CATEGORY_COUNT): wordProbByClass.append(collections.defaultdict(float))

	for i in range(nb_constants.CATEGORY_COUNT):
		for key, val in wordCountByClass[i].items():
			wordProbByClass[i][key] = (val+nb_constants.LAPLACE)/(len(trainSet)/nb_constants.CATEGORY_COUNT + 2*nb_constants.LAPLACE)

	# ignore the most common words with little semantic meaning 
	topWords = ["the", "be", "to", "of", "and", "a", "in", "that", "have", "it", "for", "not", "on", "with", "he", "as", "you", "do", "at", \
		"this", "but", "his", "by", "from", "they", "we", "say", "her", "she", "or", "will", "an", "my", "one", "all", "would", "there", "their", "what"]
	for i in range(nb_constants.CATEGORY_COUNT):
		for word in topWords:
			wordProbByClass[i][word] = 1.0

	return wordProbByClass


# takes in comment string
def classWeightsFromComment(entry, wordProbByClass):
	weightByClass = [1.0 for _ in range(nb_constants.CATEGORY_COUNT)]

	entryList = entry[nb_constants.COMMENT_INDEX].split()
	for word in entryList:
		normalizedWord = re.sub(r'[^\w\s]','',word.lower())
		for i in range(nb_constants.CATEGORY_COUNT):
			weightByClass[i] *= wordProbByClass[i][normalizedWord] if wordProbByClass[i][normalizedWord] != 0 else float(nb_constants.LAPLACE)/(nb_constants.TRAINSET_LEN/nb_constants.CATEGORY_COUNT + 2*nb_constants.LAPLACE)
	return weightByClass


def classAndEvalMulti(testSet, wordProbByClass):
	countByClass = [0 for _ in range(nb_constants.CATEGORY_COUNT)]
	correctCountByClass = [0 for _ in range(nb_constants.CATEGORY_COUNT)]
	weightByClassList = []

	for i in range(len(testSet)):
		entry = testSet[i]
		weightByClass = classWeightsFromComment(entry, wordProbByClass)

		weightByClassList.append(weightByClass)
		classification = weightByClass.index(max(weightByClass))

		countByClass[classification] += 1

		if entry[-1] == classification:
			correctCountByClass[classification] += 1

	return countByClass, correctCountByClass, weightByClassList

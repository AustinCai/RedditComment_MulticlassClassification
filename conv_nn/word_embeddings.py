'''
Creates a word embedding dictionary that, given a word, returns a vector representation
of the word. Dictionary created from word vectors pretrained by GloVe. See bottom of file 
for a usage example. 
'''
import collections
import pickle


def getWordEmbeddingDict(vocabSize = 10000):

	word_vec_dict = collections.defaultdict(int)

	with open("../../data/glove.42B.300d/glove.42B.300d.txt", 'r') as file:

		for lineNum in range(vocabSize):
			line = file.readline()
			lineList = [i for i in line.split()]

			key = lineList[0]
			val = [float(i) for i in lineList[1:]]
			word_vec_dict[key] = val

	return word_vec_dict 


word_vec_dict = getWordEmbeddingDict(100000)

with open("embeddings.pkl", 'wb') as f:
    pickle.dump(word_vec_dict, f)

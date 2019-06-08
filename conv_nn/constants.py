SMALL_DATASET = '../data/condensed_dataset_SMALL.pkl'
FULL_DATASET = '../../data/condensed_dataset.pkl'

CATEGORY_COUNT = 20 # number of total categories in dataset CHANGE
TESTSET_PROPORTION = 0.80

LAPLACE = 1
COMMENT_INDEX = 17 # CHANGE 
WORDVEC_LEN = 300


TRAINSET_LEN = 800000
TRAINSET_CAT_LEN = 40000


tinyDataset = [
	["friend!", 0],
	["friend good!", 0], 
	["I love you so much", 0],
	["do you want to see a movie?", 0],
	["good friend", 0], # test set
	["enemy!", 1],
	["enemy bad!", 1],
	["I hate you", 1],
	["get out of my sight", 1],
	["hate enemy", 1], # test set
]


smallDataset = [
	["friend!", 0],
	["friend good!", 0], 
	["I love you so much", 0],
	["do you want to see a movie?", 0],
	["you are so important to me", 0],
	["there is no one I'd rather spend time with than you", 0],
	["we should see each other more", 0],
	["I'm so happy we got closer", 0],
	["happy important good", 0],
	["I want to spend time with you so much", 0],
	["enemy!", 1],
	["enemy bad!", 1],
	["I hate you", 1],
	["get out of my sight", 1],
	["evil exists because of people like you", 1],
	["I don't want to see you", 1],
	["I've never met anyone as incompetant as you", 1],
	["can you please leave me be", 1],
	["I hate you more than anyone", 1],
	["I will never like you", 1],
]

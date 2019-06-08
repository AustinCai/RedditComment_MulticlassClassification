import sqlite3
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

# IMPORTANT - This directory path is specific to Austin's computer. You must modify it accordingly. 
sql_conn = sqlite3.connect('../data/reddit-comments-may-2015/database.sqlite')

top_20 = ['AskReddit', 'leagueoflegends', 'nba', 'funny', 'pics', 'nfl', 'pcmasterrace', \
    'videos', 'news', 'todayilearned', 'DestinyTheGame', 'worldnews', 'soccer', 'DotA2', \
    'AdviceAnimals', 'WTF', 'GlobalOffensive', 'hockey', 'movies', 'SquaredCircle']

dataset = []

for sub_index in tqdm(range(20)):
    print sub_index

    s = "SELECT * FROM May2015 WHERE LENGTH(body) > 70 AND subreddit in ('%s') LIMIT 50000" % (top_20[sub_index])
    sub_dataset_list = (pd.read_sql(s, sql_conn)).values.tolist() # list of lists

    for entry in sub_dataset_list:
        entry.append(sub_index)
        dataset.append(entry)

with open('condensed_dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)

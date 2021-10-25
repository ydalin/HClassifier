import pandas as pd
from pandas import read_csv
import tensorflow as tf
from word_list_creator import get_stats
import numpy as np
import pickle

negative_data = read_csv('negative_data_file.csv', names=['joke', 'funny'])
positive_data = read_csv('positive_data_file.csv', names=['joke', 'funny'])

negative_data['funny'] = False
positive_data['funny'] = True


#####
# REMOVE FOR Real TRAINING!!!!!!!!
# negative_data = negative_data[:10]
# positive_data = positive_data[:10]
#####

# Merge and randomize positive and negative data
data_in = pd.concat([negative_data, positive_data]).sample(frac=1)


def gather_data(data=data_in, test_split=0.2, validation_split=0.05):
    joke_stats = []
    for i in range(len(data)):
        stats = get_stats(data.iloc[i, 0])
        joke_stats.append((stats[0], data.iloc[i, 1], data.iloc[i, 0], stats[1]))
    split_data = joke_stats
    test_split = int(len(data) - test_split * len(data))
    test_data = split_data[test_split:]
    split_data = split_data[:test_split]
    validation_split = int(len(split_data) - validation_split * len(split_data))
    validation_data = split_data[validation_split:]
    train_data = split_data[:validation_split]
    return train_data, test_data, validation_data


stats = gather_data()

file_name = "stats.pkl"

open_file = open(file_name, "wb")
pickle.dump(stats, open_file)
open_file.close()

import pandas as pd
from pandas import read_csv
import tensorflow as tf
from word_list_creator import get_stats
import numpy as np

negative_data = read_csv('negative_data_file.csv', names=['joke', 'funny'])
positive_data = read_csv('positive_data_file.csv', names=['joke', 'funny'])

negative_data['funny'] = True
positive_data['funny'] = False


#####
# REMOVE FOR Real TRAINING!!!!!!!!
# negative_data = negative_data[:20]
# positive_data = positive_data[:20]
#####

# Merge and randomize positive and negative data
data_in = pd.concat([negative_data, positive_data]).sample(frac=1)


def gather_data(data=data_in, test_split=0.2, validation_split=0.05):
    joke_stats = []
    for i in range(len(data)):
        stats = get_stats(data.iloc[i, 0])
        joke_stats.append(stats)
        # joke_stats.append(data.iloc[i, 1]) #try this later, to append labels here
    joke_stats = np.array(joke_stats)
    labels = data.iloc[:, 1].to_numpy()
    final_jokes_data = np.zeros((joke_stats.shape[0], joke_stats.shape[1]+1))
    final_jokes_data[:, :joke_stats.shape[1]] = joke_stats
    final_jokes_data[:, -1] = labels
    split_data = final_jokes_data
    test_split = int(1 - test_split * len(data))
    test_data = split_data[test_split:]
    split_data = split_data[:test_split]
    validation_split = int(1 - validation_split * len(split_data))
    validation_data = split_data[validation_split:]
    train_data = split_data[:validation_split]
    return train_data, test_data, validation_data


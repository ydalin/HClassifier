import pandas as pd
from pandas import read_csv
from word_list_creator import get_stats
import pickle

def gather_data(split=0.1):
    """
    Imports raw data from .csv files, randomizes positive and negative datasets
        & converts them to train and test lists of 2d NumPy arrays
    :param split: % of data to use for test data (float)
    :returns train data, test data: (lists containing -
        [stats (2d NumPy array), joke from input data (str), funny/not funny (bool),
            joke from output data (str, should be the same as joke from input data]
    """

    negative_data = read_csv('negative_data_file.csv', names=['joke', 'funny'])
    positive_data = read_csv('positive_data_file.csv', names=['joke', 'funny'])

    data_min_length = min([negative_data.shape[0], positive_data.shape[0]])

    negative_data = negative_data[:data_min_length]
    positive_data = positive_data[:data_min_length]

    negative_data['funny'] = False
    positive_data['funny'] = True


    ########################
    # Select a subset of the data to speed up tests: REMOVE FOR REAL TRAINING!!!!!!!!
    # negative_data = negative_data[:100]
    # positive_data = positive_data[:100]
    ########################

    # Merge and randomize positive and negative data
    data = pd.concat([negative_data, positive_data]).sample(frac=1)

    joke_stats = []
    present_gathered = 0
    print(str(0) + '% Data processed')
    for i in range(len(data)):
        if (i*100/len(data))-present_gathered >= 1:
            present_gathered = int(i*100/len(data))
            print(str(int(i*100/len(data))) + '% Data processed')
        stats = get_stats(data.iloc[i, 0])
        joke_stats.append((stats[0], data.iloc[i, 1], data.iloc[i, 0], stats[1]))
    train_data = joke_stats[:split]
    test_data = joke_stats[split:]
    print('Data fully processed!')
    return train_data, test_data


def write_to_pickle(split=0.1):
    """
    Gets data, processes it, writes to a pickle file
    :param split: % of data to use for test data (float)
    """
    stats = gather_data(split)
    file_name = "stats.pkl"
    open_file = open(file_name, "wb")
    pickle.dump(stats, open_file)
    open_file.close()

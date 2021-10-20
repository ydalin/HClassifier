import pandas as pd
from pandas import read_csv
import tensorflow as tf

max_features = 10000
sequence_length = 250

# vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
#     max_tokens=max_features,
#     output_mode='int',
#     output_sequence_length=sequence_length)




negative_data = read_csv('negative_data_file.csv', names=['joke', 'funny'])
positive_data = read_csv('positive_data_file.csv', names=['joke', 'funny'])

negative_data['funny'] = 'True'
positive_data['funny'] = 'False'


data = pd.concat([negative_data, positive_data]).sample(frac=1)




def gather_data(test_split=0.1, validation_split=0.2):
    # tokenizer = Tokenizer(
    #     num_words=None,
    #     filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    #     lower=True, split=' ', char_level=False, oov_token=None
    # )

    split_data = data.copy()

    test_split = int(1 - test_split * len(split_data))
    test_data = split_data[test_split:]
    split_data = split_data[:test_split]

    validation_split = int(1 - validation_split * len(split_data))
    validation_data = split_data[validation_split:]
    train_data = split_data[:validation_split]

    train_data = tf.data.Dataset.from_tensors(train_data.iloc[:, 0]), tf.data.Dataset.from_tensors(train_data.iloc[:, 1])
    test_data = tf.data.Dataset.from_tensor_slices(test_data.iloc[:, 0]), tf.data.Dataset.from_tensor_slices(test_data.iloc[:, 1])
    validation_data = tf.data.Dataset.from_tensor_slices(validation_data.iloc[:, 0]), tf.data.Dataset.from_tensor_slices(validation_data.iloc[:, 1])

    return train_data, test_data, validation_data
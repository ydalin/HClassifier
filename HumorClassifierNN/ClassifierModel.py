import numpy as np
import pandas as pd
from keras.models import Model
from keras import layers
from keras.layers import Input
from keras.layers import Bidirectional
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import MaxPooling1D
import nltk
from nltk.corpus import wordnet as wn
from word_list_creator import get_similarities
import tensorflow as tf


def get_model(data):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=data.shape[1] - 1))
    model.add(tf.keras.layers.Dense(8, activation='relu'))
    model.add(Bidirectional(8, activation='relu'))
    model.add(tf.keras.layers.Dense(4, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='softmax'))
    return model

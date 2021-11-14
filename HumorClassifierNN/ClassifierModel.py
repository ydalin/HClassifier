import numpy as np
from keras.models import Model
from keras import layers
from keras.layers import Dense
import nltk
from nltk.corpus import wordnet as wn
from word_list_creator import get_similarities
import tensorflow as tf


def get_model(data):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(4, input_shape=(data.shape[1], data.shape[2], ), activation='relu'))
    model.add(tf.keras.layers.Dense(4))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model

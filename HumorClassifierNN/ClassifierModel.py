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
    model.add(tf.keras.layers.Dense(4, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='softmax'))
    return model



class ClassifierNNModel:
    def __init__(self, train_dataset, test_dataset, validation_dataset, split=0.8):
        self.path_sim_data = None
        self.wup_sim_data = None
        self.lch_sim_data = None
        self.model = None
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.validation_dataset = validation_dataset

        # self.train_dataset = self.train_dataset.iloc[:, 0], self.train_dataset.iloc[:, 1]
        # self.test_dataset = self.test_dataset.iloc[:, 0], self.test_dataset.iloc[:, 1]
        # self.validation_dataset = self.validation_dataset.iloc[:, 0], self.validation_dataset.iloc[:, 1]

        # print(self.train_dataset)


    def get_model(self, data):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=data.shape[1]-1))
        model.add(tf.keras.layers.Dense(8, activation='relu'))
        model.add(tf.keras.layers.Dense(4, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='softmax'))
        return model

    # def train(self):
    #     # Compile model
    #     self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #     # Fit the model
    #     target = self.train_dataset.pop()
    #     self.model.fit(self.train_dataset, self.test_dataset, validation_split=0.33, epochs=150, batch_size=10)

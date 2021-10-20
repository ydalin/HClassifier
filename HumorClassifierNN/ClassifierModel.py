import numpy as np
import pandas as pd
from keras.models import Model
from keras import layers
from keras.layers import Input
from keras.layers import TextVectorization
from keras.layers import Bidirectional
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import SimpleRNN
import nltk
from nltk.corpus import wordnet as wn
from word_list_creator import get_similarities
import tensorflow as tf


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


    def get_model(self, classifier=False):

        vectorize_layer = TextVectorization(
            max_tokens=None, standardize='lower_and_strip_punctuation',
            split='whitespace', ngrams=None, output_mode='int',
            output_sequence_length=None, pad_to_max_tokens=False)
        inputs = Input(shape=())
        # x = vectorize_layer(inputs)
        # x = layers.Embedding(input_dim=len(vectorize_layer.get_vocabulary()), output_dim=4, input_length=2)(x)

        x = layers.Embedding(input_dim=len(vectorize_layer.get_vocabulary()), output_dim=4, input_length=2)(inputs)

        # x = layers.Dropout(0.5)(x)
        # x = layers.Conv1D(128, 1, padding="valid", activation="relu")(x)
        # x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)

        model = tf.keras.Model(inputs, predictions)
        return model

    def train(self):
        # Compile model
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Fit the model
        target = self.train_dataset.pop()
        self.model.fit(self.train_dataset, self.test_dataset, validation_split=0.33, epochs=150, batch_size=10)

import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input
from keras.layers.experimental.preprocessing import TextVectorization
from keras.layers import Bidirectional
from keras.layers import Embedding
from keras.layers import Dense
import nltk
from nltk.corpus import wordnet as wn
from word_list_creator import get_similarities
import tensorflow as tf


class ClassifierNNModel:
    def __init__(self, dataset=None):
        self.words = {}
        self.jokes = {}
        self.path_sim_data = None
        self.wup_sim_data = None
        self.lch_sim_data = None
        self.dataset = dataset
        self.model = None

    def input_dataset(self, dataset):
        self.dataset = dataset

    def split_dataset(self, split=0.8):
        split_pt = int(len(self.dataset.index)*split)
        self.train_dataset, self.test = self.dataset[:split_pt], self.dataset[split_pt:]

    def get_model(self, classifier=False):
        VOCAB_SIZE = 1000
        encoder = TextVectorization(
            max_tokens=VOCAB_SIZE)
        encoder.adapt(self.train_dataset.map(lambda text, label: text))
        visible = Input(shape=(10,))
        Embedding(input_dim=len(encoder.get_vocabulary()), output_dim=64, mask_zero=True),
        hidden1 = Dense(32, activation='relu')(visible)
        hidden2 = Dense(32, activation='relu')(hidden1)
        hidden3 = Dense(32, activation='relu')(hidden2)
        output = Dense(1, activation='sigmoid')(hidden3)
        model = Model(inputs=visible, outputs=output)
        return model

    def train(self):
        # Compile model
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Fit the model
        self.model.fit(self.train_dataset, self.test, validation_split=0.33, epochs=150, batch_size=10)

import numpy as np
import pandas as pd
from keras.models import Model
from keras import layers
from keras.layers import Input
from keras.layers.experimental.preprocessing import TextVectorization
from keras.layers import Bidirectional
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import SimpleRNN
import nltk
from nltk.corpus import wordnet as wn
from word_list_creator import get_similarities
import tensorflow as tf


class ClassifierNNModel:
    def __init__(self, dataset=None, split=0.8):
        self.words = {}
        self.jokes = {}
        self.path_sim_data = None
        self.wup_sim_data = None
        self.lch_sim_data = None
        self.dataset = dataset
        self.model = None
        self.train_dataset = None
        print('dataset')
        self.dataset = dataset
        print(dataset)
        split_pt = int(len(self.dataset[0])*split)
        self.train_dataset = self.dataset[0][split_pt], self.dataset[1][split_pt]
        self.test_dataset = self.dataset[0][split_pt:], self.dataset[1][split_pt:]
        print('train dataset')
        print(self.train_dataset)

    def get_model(self, classifier=False):
        # VOCAB_SIZE = 1000
        # encoder = TextVectorization(
        #     max_tokens=VOCAB_SIZE)
        # encoder.adapt(self.train_dataset[0].astype(np.uint32))
        # print('vocabulary')
        # print(encoder.get_vocabulary())
        rnn = SimpleRNN(7)
        embedding_vectors = Embedding(input_dim=len(self.train_dataset[0]), output_dim=64, mask_zero=True)
        output_vector = rnn(embedding_vectors)
        hidden1 = Dense(32, activation='relu')(output_vector)
        hidden2 = Dense(32, activation='relu')(hidden1)
        hidden3 = Dense(32, activation='relu')(hidden2)
        output = Dense(1, activation='sigmoid')(hidden3)
        model = Model(inputs=layers.Input(shape=200), outputs=output)
        return model

    def train(self):
        # Compile model
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Fit the model
        self.model.fit(self.train_dataset, self.test_dataset, validation_split=0.33, epochs=150, batch_size=10)

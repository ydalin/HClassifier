# Based on:
# https://ai-pool.com/d/keras-get-layer-of-the-model
# https://www.kaggle.com/shanekonaung/reuters-dataset-using-keras
# https://stackoverflow.com/questions/43715047/how-do-i-get-the-weights-of-a-layer-in-keras
# https://stackoverflow.com/questions/46817085/keras-interpreting-the-output-of-get-weights
# https://stackoverflow.com/questions/42039548/how-to-check-the-weights-after-every-epoc-in-keras-model

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import reuters
# built in keras function for one-hot encoding
from keras.utils.np_utils import to_categorical
from keras.callbacks import LambdaCallback
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence] = 1
    return results

def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i,label in enumerate(labels):
        results[i,label] = 1
    return results

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

word_index = reuters.get_word_index()
reverse_word_index = dict(
                        [(value,key) for (key, value) in word_index.items()])
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])



x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(10000,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(46, activation='softmax'))

weights = []
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

x_val = x_train[:100000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:100000]
partial_y_train = one_hot_train_labels[1000:]

get_weights = LambdaCallback(on_batch_end=lambda batch, logs: weights.append(model.get_weights()))
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=1,
                    batch_size=300,
                    validation_data=(x_val,y_val),
                    callbacks=[get_weights])

first_layer_avg_weights = []
first_layer_avg_biases = []
seventh_layer_avg_weights = []
seventh_layer_avg_biases = []
for i in range(len(weights)):
    epoch = weights[i]
    first_layer_avg_weights.append(np.average(epoch[0]))
    first_layer_avg_biases.append(np.average(epoch[1]))
    seventh_layer_avg_weights.append(np.average(epoch[13]))
    seventh_layer_avg_biases.append(np.average(epoch[14]))

kernel = [-1, 1]
first_layer_avg_biases = np.abs(np.convolve(first_layer_avg_biases, kernel))
seventh_layer_avg_biases = np.abs(np.convolve(seventh_layer_avg_biases, kernel))
first_layer_avg_weights = np.abs(np.convolve(first_layer_avg_weights, kernel))
seventh_layer_avg_weights = np.abs(np.convolve(seventh_layer_avg_weights, kernel))
print(first_layer_avg_biases)
print(seventh_layer_avg_biases)

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

# plt.plot(epochs, loss, 'r', label='Training Loss')
# plt.plot(epochs, val_loss, 'b', label='Validation Loss')
# plt.plot(epochs, first_layer_avg_biases[:-1], label='first layer change in biases')
# plt.plot(epochs, seventh_layer_avg_biases[:-1], label='seventh layer change in biases')
x = []
for i in range(len(first_layer_avg_weights)-1):
    x.append(i+1)
plt.subplot(2, 1, 1)
plt.plot(x, first_layer_avg_biases[:-1], label='1st layer, delta biases')
plt.plot(x, seventh_layer_avg_biases[:-1], label='7th layer, delta biases')

plt.subplot(2, 1, 2)
plt.plot(x, first_layer_avg_weights[:-1], label='1st layer, delta weight')
plt.plot(x, seventh_layer_avg_weights[:-1], label='7th layer, delta weight')
# plt.title('Change in Weights')
plt.xlabel('Training Batches')
plt.ylabel('Change')
plt.legend()
plt.savefig('Change_in_weights.png')
plt.show()

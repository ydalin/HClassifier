import numpy as np
# from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.callbacks import LambdaCallback
from keras.utils.np_utils import to_categorical
import nltk
from nltk.corpus import wordnet as wn

for word in wn.words():
    print(word)
    break


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1
    return results


visible = Input(shape=(10,))
hidden1 = Dense(10, activation='relu')(visible)
hidden2 = Dense(20, activation='relu')(hidden1)
hidden3 = Dense(20, activation='relu')(hidden2)
hidden4 = Dense(20, activation='relu')(hidden3)
hidden5 = Dense(20, activation='relu')(hidden4)
hidden6 = Dense(10, activation='relu')(hidden5)
output = Dense(1, activation='sigmoid')(hidden6)
model = Model(inputs=visible, outputs=output)

weights = []
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# x_train = vectorize_sequences(train_data)
# x_test = vectorize_sequences(test_data)
#
# one_hot_train_labels = to_categorical(train_labels)
# one_hot_test_labels = to_categorical(test_labels)

# x_val = x_train[:100000]
# partial_x_train = x_train[1000:]
#
# y_val = one_hot_train_labels[:100000]
# partial_y_train = one_hot_train_labels[1000:]
#
#
# get_weights = LambdaCallback(on_batch_end=lambda batch, logs: weights.append(model.get_weights()))
# history = model.fit(partial_x_train,
#                     partial_y_train,
#                     epochs=1,
#                     batch_size=300,
#                     validation_data=(x_val,y_val),
#                     callbacks=[get_weights])

# summarize layers
print(model.summary())
# plot graph
# plot_model(model, to_file='multilayer_perceptron_graph.png')
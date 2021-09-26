import keras
from keras import layers
from keras.layers import Conv1D, Lambda, Dense
import tensorflow as tf

def summation(x):
    y = tf.reduce_sum(x, 0)
    return y


class ANN:
    def __init__(self, num_layers=7, input_shape=(3, ), layer_size=64):
        self.branches = 1
        self.input_shape = input_shape
        inputs = keras.Input(shape=input_shape, name='data')
        x = Dense(64, activation="relu")(inputs)
        for i in range(num_layers):
            x = Dense(64, activation="relu")(x)
        outputs = Lambda(summation)(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs, name="A-NN")

    def get_model(self):
        return self.model

    def branch(self, root_layer=-1):
        """
        clones the model, with a new parallel layer; layers before the root layer are shared, the root layer and
        subsequent layers are not
        :param root_layer: first not shared layer
        :return: branched model
        """
        self.branches += 1
        inputs = keras.Input(shape=input_shape, name='data')
        x = Dense(64, activation="relu")(inputs)
        for layer in range(len(self.model.layers)+root_layer):

        return branched_model

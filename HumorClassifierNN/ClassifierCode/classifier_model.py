import tensorflow as tf
import numpy as np

def get_model(data):
    """
    Create a Neural Network model
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=((data.shape[1], data.shape[2]))))
    model.add(tf.keras.layers.Dense(9, activation='relu'))
    model.add(tf.keras.layers.Dense(9, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model
# model = get_model(np.zeros((200, 3, 3)))
# model.summary()
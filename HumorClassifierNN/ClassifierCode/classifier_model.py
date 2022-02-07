import tensorflow as tf


def get_model(data):
    """
    Create a Neural Network model
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(8, input_shape=(data.shape[1], data.shape[2], ), activation='relu'))
    model.add(tf.keras.layers.Dense(8))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model

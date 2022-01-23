import tensorflow as tf


def get_model(data):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(9, input_shape=(data.shape[1], data.shape[2], ), activation='relu'))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model

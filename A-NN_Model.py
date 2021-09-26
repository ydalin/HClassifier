import keras
from keras import layers
from keras.layers import Dense
class ANN:
    def __init__(self, num_layers=7, input_shape=2, layer_size=64):
        self.branches = 1
        inputs = keras.Input(input_shape)
        dense = Dense(64, activation="relu")
        x = dense(inputs)
        x = Dense(64, activation="relu")(x)
        outputs = Dense(10)(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs, name="A-NN")
        for i in range(num_layers):
            self.model.add(Dense(layer_size, activation='relu'))

    def get_model(self):
        return self.model
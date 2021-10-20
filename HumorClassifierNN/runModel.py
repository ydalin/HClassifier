import numpy as np
from ClassifierModel import ClassifierNNModel as ClassNN
# from word_list_creator import results
from gather_data import gather_data
import tensorflow as tf


train_data, test_data, validation_data = gather_data()
# noun_sims, verb_sims, word_list = results


model = ClassNN(train_data, test_data, validation_data).get_model(train_data)

# class_model.compile()
# class_model.fit(x=train_data[:, :-1], y=train_data[:, -1], epochs=150, batch_size=10)

x = train_data[:, :-1]
y = train_data[:, -1]
x_test = test_data[:, :-1]
y_test = test_data[:, -1]

model.compile(optimizer='sgd', loss='mse', metrics=['mse', 'acc'])
history = model.fit(x, y, batch_size=4, epochs=3)

print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=4)
print("test loss, mean squared error, test accuracy:", results)

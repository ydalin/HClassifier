import numpy as np
from ClassifierModel import get_model
# from word_list_creator import results
from gather_data import gather_data
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau

train_data, test_data, validation_data = gather_data()

model = get_model(train_data)

x = train_data[:, :-1]
y = train_data[:, -1].astype(int)
x_test = test_data[:, :-1]
y_test = test_data[:, -1].astype(int)
x_val = validation_data[:, :-1]
y_val = validation_data[:, -1].astype(int)

model.compile(optimizer='adam', loss='mse', metrics=['mse', 'acc'])
history = model.fit(x, y, batch_size=500, epochs=5)
print(history.history)
print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=500)
print("test loss, mean squared error, test accuracy:", results)

num_predictions = 200
# make sure num_predictions is < len(x_val)
num_predictions = num_predictions*int(len(x_val) > num_predictions) + int(len(x_val) <= num_predictions)*len(x_val)

predictions = model.predict(x_val[:num_predictions])

true = []
false = []
for i in range(len(predictions)):
    if predictions[i][0] > 0.7:
        true.append((str(bool(predictions[i][0])), y_val[i]))
    elif predictions[i][0] < 0.4:
        false.append((str(bool(predictions[i][0])), y_val[i]))
    print('prediction: ' + str(bool(predictions[i][0])) + ' correct answer: ' + str(bool(y_val[i])))

print('Predicted True: ' + str(len(true)))
for i in true:
    print('prediction: ' + true[i] + ' correct answer: ' + str(bool(true[i][1])))

print('Predicted False: ' + str(len(false)))
for i in false:
    print('prediction: ' + true[i] + ' correct answer: ' + str(bool(true[i][1])))

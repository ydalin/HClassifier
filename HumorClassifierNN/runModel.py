import numpy as np
from ClassifierModel import get_model
# from word_list_creator import results
from gather_data import gather_data
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau

train_data, test_data, validation_data = gather_data()

x = []
y = []
j = []
print(train_data)
for i in range(len(train_data)):
    d = train_data[i]
    print(d[1])
    x.append(d[0])
    y.append(d[1])
    j.append(d[2])
x_test = []
y_test = []
j_test = []
for i in range(len(test_data)):
    d = test_data[i]
    x_test.append(d[0])
    y_test.append(d[1])
    j_test.append(d[2])
x_val = []
y_val = []
j_val = []

for i in range(len(validation_data)):
    d = validation_data[i]
    x_val.append(d[0])
    y_val.append(d[1])
    j_val.append(d[2])

x = np.array(x)
y = np.array(y).astype(int)
x_test = np.array(x_test)
y_test = np.array(y_test).astype(int)
x_val = np.array(x_val)
y_val = np.array(y_val).astype(int)

model = get_model(x)

model.compile(optimizer='adam', loss='mse', metrics=['mse', 'acc'])
history = model.fit(x, y, batch_size=500, epochs=3)
print('history:')
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
for i in range(len(true)):
    print('prediction: ' + true[i][0] + ' correct answer: ' + str(bool(true[i][1])))

print('Predicted False: ' + str(len(false)))
for i in range(len(false)):
    print('prediction: ' + false[i][0] + ' correct answer: ' + str(bool(false[i][1])))

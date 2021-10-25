import numpy as np
from ClassifierModel import get_model
# from word_list_creator import results
# from gather_data import gather_data
import tensorflow as tf
# from keras.callbacks import ReduceLROnPlateau
import pickle

print('Gathering data')

file_name = "stats.pkl"
open_file = open(file_name, "rb")
stats = pickle.load(open_file)
open_file.close()

train_data, test_data, validation_data = stats

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
    print(d)
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

print('training')
print(x.shape)
model = get_model(x.shape)

model.compile(optimizer='adam', loss='mse', metrics=['mse', 'acc'])
history = model.fit(x, y, epochs=150)
# print('history:')
# print(history.history)
print("Evaluate on test data")
results = model.evaluate(x_test, y_test)
# print("test loss, mean squared error, test accuracy:", results)

num_predictions = 2000
# make sure num_predictions is < len(x_val)
num_predictions = num_predictions*int(len(x_val) > num_predictions) + int(len(x_val) <= num_predictions)*len(x_val)

predictions = model.predict(x_val[:num_predictions])

true = []
false = []
for i in range(len(predictions)):
    if predictions[i][0] > 1:
        true.append((predictions[i][0], y_val[i]))
    elif predictions[i][0] < 0.1:
        false.append((predictions[i][0], y_val[i]))
    # print('prediction: ' + str(predictions[i][0]) + ', correct answer: ' + str(bool(y_val[i])) + ', joke: ' + str(j_val[i]) + '\n')

count_correct_true = 0
count_correct_false = 0
print('Predicted True: ' + str(len(true)))
for i in range(len(true)):
    # print('prediction: ' + str(true[i][0]) + ', correct answer: ' + str(bool(y_val[i])) + ', joke: ' + str(j_val[i]) + '\n')
    if bool(y_val[i]) == True:
        count_correct_true += 1

print('Predicted False: ' + str(len(false)))
for i in range(len(false)):
    # print('prediction: ' + str(false[i][0]) + ', correct answer: ' + str(bool(y_val[i])) + ', joke: ' + str(j_val[i]) + '\n')
    if bool(y_val[i]) == False:
        count_correct_false += 1

print('total correct true: ' + str(count_correct_true) + ', pct correct: ' + str(count_correct_true*100/len(true)))
print('total correct false: ' + str(count_correct_false) + ', pct correct: ' + str(count_correct_false*100/len(false)))
false_count = np.where(y_val.copy() == False)[0].shape[0]
true_count = np.where(y_val.copy() == True)[0].shape[0]

print(false_count/(false_count+true_count), true_count/(false_count+true_count))

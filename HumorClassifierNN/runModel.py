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

# stats = gather_data()

print('Data Gathered')
train_data, test_data, validation_data = stats

x = []
y = []
j = []
for i in range(len(train_data)):
    d = train_data[i]
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
x = np.asarray(x).astype('float32')
y = np.asarray(y).astype(int)
x_test = np.asarray(x_test).astype('float32')
y_test = np.asarray(y_test).astype(int)
x_val = np.asarray(x_val).astype('float32')
y_val = np.asarray(y_val).astype(int)

print('training')
model = get_model(x)

model.compile(optimizer='adam', loss='mse', metrics=['mse', 'acc'])
history = model.fit(x, y, epochs=30)
# print('history:')
# print(history.history)
print("Evaluate on test data")
results = model.evaluate(x_test, y_test)
# print("test loss, mean squared error, test accuracy:", results)

num_predictions = len(x_val)
# make sure num_predictions is < len(x_val)
num_predictions = num_predictions*int(len(x_val) > num_predictions) + int(len(x_val) <= num_predictions)*len(x_val)

predictions = model.predict(x_val[:num_predictions])

true = []
false = []
print(predictions[0])
predictions = sorted(predictions, key=lambda x: np.max(x), reverse=True)
print(predictions[:20])

slice = 5
for i in range(slice):
    true.append((predictions[i], y_val[i]))
for i in range(len(predictions)-slice, len(predictions)):
    false.append((predictions[i], y_val[i]))
    # print('prediction: ' + str(predictions[i][0]) + ', correct answer: ' + str(bool(y_val[i])) + ', joke: ' + str(j_val[i]) + '\n')

count_correct_true = 0
count_correct_false = 0
print('True: ' + str(len(true)))
for i in range(len(true)):
    # print('prediction: ' + str(true[i][0]) + ', correct answer: ' + str(bool(y_val[i])) + ', joke: ' + str(j_val[i]) + '\n')
    if bool(y_val[i]) == True:
        count_correct_true += 1

print('False: ' + str(len(false)))
for i in range(len(false)):
    # print('prediction: ' + str(false[i][0]) + ', correct answer: ' + str(bool(y_val[i])) + ', joke: ' + str(j_val[i]) + '\n')
    if bool(y_val[i]) == False:
        count_correct_false += 1

print('total correct true: ' + str(count_correct_true) + ', total True: ' + str(len(true)) + ', pct correct: ' + str(count_correct_true*100/(len(true)+.0001)))
print('total correct false: ' + str(count_correct_false) + ', total False: ' + str(len(false)) + ', pct correct: ' + str(count_correct_false*100/(len(false)+.0001)))
false_count = np.where(y_val.copy() == False)[0].shape[0]
true_count = np.where(y_val.copy() == True)[0].shape[0]

# for i in range(predictions.shape[0]):
#     print(predictions[i])
#     print(y_val[i])
#     print(j_val[i])
#     print('--------------------\n')
best_guess_pos = np.where(predictions == np.amax(predictions))[0][0]
print('best guess for True: ' + str(predictions[best_guess_pos]) + ', actual answer: ' + str(y_val[best_guess_pos]) + ', joke: ' + str(j_val[best_guess_pos]))
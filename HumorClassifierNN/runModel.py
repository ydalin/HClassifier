import numpy as np
from ClassifierModel import get_model
# from word_list_creator import results
# from gather_data import gather_data
import tensorflow as tf
# from keras.callbacks import ReduceLROnPlateau
import pickle
from matplotlib import pyplot as plt

print('Gathering data')

file_name = "stats.pkl"
open_file = open(file_name, "rb")
stats = pickle.load(open_file)
open_file.close()

# stats = gather_data()

print('Data Gathered')
train_data, validation_data = stats

x = []
y = []
j = []
for i in range(len(train_data)):
    d = train_data[i]
    x.append(d[0])
    y.append(d[1])
    j.append(d[2])

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
x_val = np.asarray(x_val).astype('float32')
y_val = np.asarray(y_val).astype(int)

print('training')
model = get_model(x)

model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['acc', tf.keras.metrics.SparseCategoricalAccuracy(
    name="sparse_categorical_accuracy", dtype=None
)])
history = model.fit(x, y, epochs=250)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['sparse_categorical_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
print("Evaluate on test data")
results = model.evaluate(x_val, y_val)
print("test loss, mean squared error, test accuracy: ", results)

num_predictions = len(x_val)
# make sure num_predictions is < len(x_val)
num_predictions = num_predictions*int(len(x_val) > num_predictions) + int(len(x_val) <= num_predictions)*len(x_val)

predictions = model.predict(x_val[:num_predictions])

predictions = np.mean(predictions, axis=1)

sorted_predictions = []

for i in range(len(predictions)):
    sorted_predictions.append((predictions[i], y_val[i], j_val[i]))

sorted_predictions = sorted(sorted_predictions, key=lambda z: z[0])

slice = len(sorted_predictions)//2
slice = 5

true = sorted_predictions[len(sorted_predictions)-slice:len(sorted_predictions)]
false = sorted_predictions[:slice]

count_correct_true = 0
print('True: ' + str(len(true)))
for i in range(len(true)):
    # print('prediction: ' + str(true[i][0]) + ', correct answer: ' + str(bool(true[i])) + ', joke: ' + str(true[i]) + '\n')
    if bool(true[i][1]) == True:
        count_correct_true += 1

count_correct_false = 0
print('False: ' + str(len(false)))
for i in range(len(false)):
    # print('prediction: ' + str(false[i][0]) + ', correct answer: ' + str(bool(false[i])) + ', joke: ' + str(false[i]) + '\n')
    if bool(false[i][1]) == False:
        count_correct_false += 1

print('total correct true: ' + str(count_correct_true) + ', total True: ' + str(len(true)) + ', pct correct: ' + str(count_correct_true*100/(len(true))))
print('total correct false: ' + str(count_correct_false) + ', total False: ' + str(len(false)) + ', pct correct: ' + str(count_correct_false*100/(len(false))))

best_guess_True = max(true, key=lambda x: x[0])
print('best guess for True: ' + str(best_guess_True[0]) + ', actual answer: ' + str(best_guess_True[1]) + ', joke: ' + str(best_guess_True[2]))
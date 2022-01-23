import numpy as np
from ClassifierModel import get_model
# from word_list_creator import results
# from gather_data import gather_data
import tensorflow as tf
import pickle
from matplotlib import pyplot as plt
import seaborn as sns

print('Gathering data')

file_name = "stats.pkl"
open_file = open(file_name, "rb")
stats = pickle.load(open_file)
open_file.close()

# stats = gather_data()

print('Data Gathered')
train_data, test_data = stats

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


x = np.asarray(x).astype('float32')
y = np.asarray(y).astype(int)
x_test = np.asarray(x_test).astype('float32')
y_test = np.asarray(y_test).astype(int)

print('training')
model = get_model(x)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', tf.keras.metrics.BinaryCrossentropy(
    name="binary_crossentropy", dtype=None)])
history = model.fit(x, y, validation_split=0.2, epochs=5)
model.save("stats_model")
# Plot training & validation accuracy values
# plt.plot(history.history['acc'])
# plt.plot(history.history['sparse_categorical_accuracy'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
# print("Evaluate on test data")
# results = model.evaluate(x_test, y_test)
# print("test loss, mean squared error, test accuracy: ", results)

# num_predictions = len(x_val)
# # make sure num_predictions is < len(x_val)
# num_predictions = num_predictions*int(len(x_val) > num_predictions) + int(len(x_val) <= num_predictions)*len(x_val)
#




predictions = model.predict(x_test)
predictions = np.mean(predictions, axis=1).transpose()[0]
predictions = np.vstack([predictions, ((predictions >= 0.5).astype(int)==y_test).astype(int)])
# correct = predictions[:, predictions[1]==1][0]
# incorrect = predictions[:, predictions[1]==0][0]

# sns.distplot(predictions[0], color='green', label='correct')
sns.kdeplot(predictions[0], predictions[1], fill=True)
plt.legend()
plt.show()
# bins = 200
# plt.hist(correct, density=True, bins=bins, label='correct', color='green')
# plt.hist(incorrect, density=True, bins=bins, label='incorrect', color='red')
# plt.legend()
# plt.show()


# predictions = np.mean(predictions, axis=1)
#
# sorted_predictions = []
#
# for i in range(len(predictions)):
#     sorted_predictions.append((predictions[i], y_val[i], j_val[i]))
#
# sorted_predictions = sorted(sorted_predictions, key=lambda z: z[0])
#
# slice = len(sorted_predictions)//2
# slice = 10
#
# true = sorted_predictions[len(sorted_predictions)-slice:len(sorted_predictions)]
# false = sorted_predictions[:slice]
#
# count_correct_true = 0
# print('True: ' + str(len(true)))
# for i in range(len(true)):
#     # print('prediction: ' + str(true[i][0]) + ', correct answer: ' + str(bool(true[i])) + ', joke: ' + str(true[i]) + '\n')
#     if bool(true[i][1]) == True:
#         count_correct_true += 1
#
# count_correct_false = 0
# print('False: ' + str(len(false)))
# for i in range(len(false)):
#     # print('prediction: ' + str(false[i][0]) + ', correct answer: ' + str(bool(false[i])) + ', joke: ' + str(false[i]) + '\n')
#     if bool(false[i][1]) == False:
#         count_correct_false += 1
#
# print('total correct true: ' + str(count_correct_true) + ', total True: ' + str(len(true)) + ', pct correct: ' + str(count_correct_true*100/(len(true))))
# print('total correct false: ' + str(count_correct_false) + ', total False: ' + str(len(false)) + ', pct correct: ' + str(count_correct_false*100/(len(false))))
#
# best_guess_True = max(true, key=lambda x: x[0])
# print('best guess for True: ' + str(best_guess_True[0]) + ', actual answer: ' + str(best_guess_True[1]) + ', joke: ' + str(best_guess_True[2]))
import numpy as np
from ClassifierModel import ClassifierNNModel as ClassNN
# from word_list_creator import results
from gather_data import gather_data
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau

train_data, test_data, validation_data = gather_data()

model = ClassNN(train_data, test_data, validation_data).get_model(train_data)

# class_model.compile()
# class_model.fit(x=train_data[:, :-1], y=train_data[:, -1], epochs=150, batch_size=10)

x = train_data[:, :-1]
y = train_data[:, -1].astype(int)
x_test = test_data[:, :-1]
y_test = test_data[:, -1].astype(int)
x_val = validation_data[:, :-1]
y_val = validation_data[:, -1].astype(int)

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.4,
                              patience=0, min_lr=0.1)

model.compile(optimizer='sgd', loss='mse', metrics=['mse', 'acc'])
history = model.fit(x, y, batch_size=500, epochs=3, callbacks=[reduce_lr])
print(history.history)
print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=500, callbacks=[reduce_lr])
print("test loss, mean squared error, test accuracy:", results)

num_predictions = 200
# make sure num_predictions is < len(x_val)
num_predictions = num_predictions*int(len(x_val) > num_predictions) + int(len(x_val) <= num_predictions)*len(x_val)

predictions = model.predict(x_val[:num_predictions], callbacks=[reduce_lr])

for i in range(len(predictions)):
    true = []
    false = []
    if predictions[i][0] > 0.7:
        true.append((predictions[i][0], y_val[i]))
    elif predictions[i][0] < 0.4:
        false.append((predictions[i][0], y_val[i]))
    print('prediction: ' + str(predictions[i][0]) + ' correct: ' + str(bool(y_val[i])))

print('Predicted True: ' + str(len(true)))
for i in true:
    print('prediction: ' + str(predictions[i][0]) + ' correct: ' + str(bool(y_val[i][1])))

print('Predicted False: ' + str(len(false)))
for i in false:
    print('prediction: ' + str(predictions[i][0]) + ' correct: ' + str(bool(y_val[i][1])))

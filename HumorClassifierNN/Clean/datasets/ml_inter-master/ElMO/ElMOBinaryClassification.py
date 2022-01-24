import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from sklearn import preprocessing
import keras
import numpy as np

'''
loading ElMO model from tensorflow hub
using default ElMO vectors, which is a mean pooling
of all the vectors into a sentence representation that 
is the average of all the individual words
'''
url = "https://tfhub.dev/google/elmo/3"
embed = hub.Module(url)

data = pd.read_csv('puns.csv', encoding='latin-1')

# title for each column of data in csv file
y = list(data['label'])
x = list(data['text'])

'''
 Using label encoder to fit our labels to get
 classes of "-1" and "1" to numeric labels, 
 representing nonjoke and joke respectively
'''
le = preprocessing.LabelEncoder()
le.fit(y)

# turn an integer to 1-hot vectors
def encode(le, labels):
    enc = le.transform(labels)
    return keras.utils.to_categorical(enc)

# turns a 1-hot vector to integer
def decode(le, one_hot):
    dec = np.argmax(one_hot, axis=1)
    return le.inverse_transform(dec)



# keep x the same, but want to decode the y label
x_enc = x
y_enc = encode(le, y)
#decrease the second dimension by 1 so the shape matches that of x_enc
#it was always outputing 0 for the last shape of the tensor
y_enc = y_enc[:, :2]

'''
selecting testing and training samples

traing with the first 3500 lines of data
72.5% of the total data set in csv
'''
x_train = np.asarray(x_enc[:3500])
y_train = np.asarray(y_enc[:3500])

# checking shape of the tensors
# print('x_train:', x_train[:2])
# print('y_train:', y_train[:2])
# print('x_train.shape:', x_train.shape)
# print('y_train.shape:', y_train.shape)

#test with rest of the data
x_test = np.asarray(x_enc[3500:])
y_test = np.asarray(y_enc[3500:])

# checking the shape of the tensors
# print('x_test.shape:', x_test.shape)
# print('y_test.shape:', y_test.shape)
# print("x_test:", x_test[:1])

# Define layers.
from keras.layers import Input, Lambda, Dense
# We want functional model instead of sequential model
from keras.models import Model
import keras.backend as K

'''
Args:
-   x: string of sentence

converting x into tensorflow string, squeeze the input 
(removing dimensions of size 1 from the shape of a tensor),
and use it to create an embedding layer

signature is default, as we are using regular string sentences
getting "default" output, as it will do an average of our individual words to
create a sentence representation
'''
def ELMoEmbedding(x):
    return embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]

# shape is 1 then the batch size, so we are taking one sentence at a time
# can play around with more number of sentences at a time.
input_text = Input(shape=(1,), dtype=tf.string)

#ElMO vectors are size 1024
embedding = Lambda(ELMoEmbedding, output_shape=(1024, ))(input_text)

# feeding embedding to dense layer with relu activation
dense = Dense(256, activation='relu')(embedding)

# feed dense layer to prediction layer
# using softmax, since there are only 2 classes (binary classification)
pred = Dense(2, activation='softmax')(dense)

# define functional model
model = Model(inputs=[input_text], outputs=pred)

# compile the model with categorical cross entropy
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# training and saving the model as elmo-model.h5
with tf.Session() as session:
    K.set_session(session)
    session.run(tf.global_variables_initializer())  
    session.run(tf.tables_initializer())
    #can play around with different epochs and batch_size
    history = model.fit(x_train, y_train, epochs=1, batch_size=32)
    model.save_weights('./elmo-model.h5')

# testing
with tf.Session() as session:
    K.set_session(session)
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    model.load_weights('./elmo-model.h5')  
    predicts = model.predict(x_test, batch_size=32)

y_test = decode(le, y_test)
print('test:', y_test)
y_preds = decode(le, predicts)
print('prediction:', y_preds)

# confusion matrix
from sklearn import metrics

print(metrics.confusion_matrix(y_test, y_preds))

print(metrics.classification_report(y_test, y_preds))

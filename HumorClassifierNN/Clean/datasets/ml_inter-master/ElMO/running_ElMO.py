import tensorflow as tf
from keras.layers import Input, Lambda, Dense
import keras.backend as K
from keras.models import Model
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from sklearn import preprocessing
import keras
import numpy as np


input = "There is an FBI wing of money laundering You could report there"

embed = hub.Module("https://tfhub.dev/google/elmo/3")

# input_csv = 'test.csv'
# data = pd.read_csv(input_csv, encoding='latin-1')
# y = list(data['label'])
# x = list(data['text'])

le = preprocessing.LabelEncoder()
le.fit([-1, 1])

def decode(le, one_hot):
    dec = np.argmax(one_hot, axis=1)
    return le.inverse_transform(dec)

input_jokes = np.asarray(["this is a joke", input])

def ELMoEmbedding(x):
    return embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]

input_text = Input(shape=(1,), dtype=tf.string)
embedding = Lambda(ELMoEmbedding, output_shape=(1024, ))(input_text)
dense = Dense(256, activation='relu')(embedding)
pred = Dense(2, activation='softmax')(dense)
model = Model(inputs=[input_text], outputs=pred)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

with tf.Session() as session:
    K.set_session(session)
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    model.load_weights('./elmo-model.h5')  
    predicts = model.predict(input_jokes, batch_size=None)

print('joke:',input_jokes[1:])

y_preds = decode(le, predicts[1:])
print('prediction:', y_preds)

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import rnn
import math
import datetime
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
max_features = 10000
maxlen = 450
embedding_size = 128
batch_size = 512
hidden_size = 300
num_epochs = 20
max_learning_rate = 0.005
min_learning_rate = 0.0001
decay_coefficient = 2.5
dropout_keep_prob = 0.5
evaluate_every = 100

train = pd.read_csv("positive_data_file.csv", header=0,delimiter="\t", quoting=3)
test = pd.read_csv("shortjokes.csv",header=0,delimiter="\t", quoting=3)

tokenizer = Tokenizer(num_words=max_features,lower=True)
tokenizer.fit_on_texts(list(train['review']) + list(test['review']))
word_index = tokenizer.word_index
x_train = tokenizer.texts_to_sequences(list(train['review']))
x_train = pad_sequences(x_train,maxlen=maxlen)
y_train = to_categorical(list(train['sentiment']))
x_test = tokenizer.texts_to_sequences(list(test['review']))
x_test = pad_sequences(x_test,maxlen=maxlen)

x_train,x_dev,y_train,y_dev = train_test_split(x_train,y_train,test_size=0.3,random_state=0)
class TextHAN(object):
    def __init__(self,
                 sequence_length,
                 num_sentences,
                 num_classes,
                 vocab_size,
                 embedding_size,
                 hidden_size,
                 batch_size,
                 l2_reg_lambda=0.0):
        initializer = tf.initializers.random_normal(stddev=0.1)


        self.input_x = tf.placeholder(tf.int32, [None,sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None,num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        sequence_length = int(sequence_length / num_sentences)

        l2_loss = tf.constant(0.0)

        input_x = tf.split(self.input_x, num_sentences, axis=1)
        input_x = tf.stack(input_x, axis=1)

        with tf.name_scope('embedding'):
            self.W = tf.Variable(tf.random_uniform([vocab_size,embedding_size],-1.0,1.0),
                                name='W',trainable=True)
            # [batch_size,num_sequences,sequence_length,embedding_size]
            self.embedded_words = tf.nn.embedding_lookup(self.W, input_x)
            # [batch_size*num_sequences,sequence_length,embedding_size]
            embedded_words_reshaped = tf.reshape(self.embedded_words,
                                                 shape=[-1, sequence_length, embedding_size])

        with tf.name_scope('word_gru'):
            gru_fw_cell = tf.keras.layers.GRUCell(hidden_size)
            gru_bw_cell = tf.keras.layers.GRUCell(hidden_size)
            embedded_words_list = tf.unstack(embedded_words_reshaped,sequence_length,axis=1)
            w_outputs, _, _ = rnn.static_bidirectional_rnn(gru_fw_cell,
                                                           gru_bw_cell,
                                                           embedded_words_list,
                                                           dtype=tf.float32)
            # [batch_size*num_sequence,sequence_length,hidden_size*2]
            w_outputs = tf.stack(w_outputs,axis=1)

        with tf.name_scope('word_attention'):
            W_word = tf.get_variable(shape=[hidden_size*2, hidden_size*2],
                                     initializer=initializer,
                                     name='W_word')
            b_word = tf.get_variable(shape=[hidden_size*2],
                                     initializer=initializer,
                                     name='b_word')
            context_vector_word = tf.get_variable("what_is_the_informative_word",
                                                  shape=[hidden_size * 2],
                                                  initializer=initializer)
            # [batch_size*num_sequence*sequence_length,hidden_size*2]
            w_hidden_state = tf.reshape(w_outputs,[-1,hidden_size*2])
            w_hidden_representation = tf.nn.tanh(tf.matmul(w_hidden_state,W_word) + b_word)
            w_hidden_representation = tf.reshape(w_hidden_representation, shape=[-1,sequence_length,hidden_size * 2])
            # [batch_size*num_sequence, sequence_length, hidden_size*2]
            w_context_similiarity = tf.multiply(w_hidden_representation,context_vector_word)
            # [batch_size*num_sequence, sequence_length]
            w_attention_logits = tf.reduce_sum(w_context_similiarity,axis=2)

            w_attention_logits_max = tf.reduce_max(w_attention_logits, axis=1,keepdims=True)
            w_p_attention = tf.nn.softmax(w_attention_logits - w_attention_logits_max)
            # [batch_size*num_sequence, sequence_length, 1]
            w_p_attention_expanded = tf.expand_dims(w_p_attention, axis=2)

            # [batch_size*num_sentences, sequence_length, hidden_size*2]
            sentence_representation = tf.multiply(w_p_attention_expanded,w_outputs)
            # [batch_size*num_sentences, hidden_size*2]
            sentence_representation = tf.reduce_sum(sentence_representation,axis=1)
            sentence_representation = tf.reshape(sentence_representation,[-1,num_sentences,hidden_size*2])

        with tf.name_scope('sentence_gru'):
            gru_fw_cell = tf.keras.layers.GRUCell(hidden_size)
            gru_bw_cell = tf.keras.layers.GRUCell(hidden_size)

            sentence_list = tf.unstack(sentence_representation,num_sentences,axis=1)
            s_outputs, _, _ = rnn.static_bidirectional_rnn(gru_fw_cell,
                                                           gru_bw_cell,
                                                           sentence_list,
                                                           dtype=tf.float32)
            # [batch_size,num_sentences,hidden_size*2]
            s_outputs = tf.stack(s_outputs,axis=1)

        with tf.name_scope('sentence_attention'):
            W_sentence = tf.get_variable(shape=[hidden_size*2, hidden_size*2],
                                         initializer=initializer,
                                         name='W_s_attention')
            b_sentence = tf.get_variable(shape=[hidden_size*2],
                                         initializer=initializer,
                                         name='b_sentence')
            context_vector_sentence = tf.get_variable("what_is_the_informative_sentence",
                                                      shape=[hidden_size * 2],
                                                      initializer=initializer)
            # [batch_size*num_sentences, hidden_size*2]
            s_hidden_state = tf.reshape(s_outputs,[-1, hidden_size*2])
            s_hidden_representation = tf.nn.tanh(tf.matmul(s_hidden_state,W_sentence) + b_sentence)
            s_hidden_representation = tf.reshape(s_hidden_representation,shape=[-1,num_sentences,hidden_size*2])
            # [batch_size,num_sentences, hidden_size*2]
            s_context_similiarity = tf.multiply(s_hidden_representation, context_vector_sentence)
            # [batch_size,num_sentences]
            s_attention_logits = tf.reduce_sum(s_context_similiarity,axis=2)
            s_attention_logits_max = tf.reduce_max(s_attention_logits, axis=1,keepdims=True)
            s_p_attention = tf.nn.softmax(s_attention_logits - s_attention_logits_max)
            # [batch_size, num_sequence, 1]
            s_p_attention_expanded = tf.expand_dims(s_p_attention, axis=2)

            # [batch_size, num_sentences, hidden_size*2]
            document_representation = tf.multiply(s_p_attention_expanded,s_outputs)
            # [batch_size, hidden_size*2]
            document_representation = tf.reduce_sum(document_representation,axis=1)
            self.output = document_representation

        with tf.name_scope('dropout'):

            self.h_drop = tf.nn.dropout(self.output,self.dropout_keep_prob)

        with tf.name_scope('output'):
            W = tf.get_variable(shape=[hidden_size*2,num_classes],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                name='W')
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        with tf.name_scope('loss'):

            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)

            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data_size = len(data)
    num_batches_per_epoch = data_size// batch_size
    for epoch in range(num_epochs):

        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch+1):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=False)
    sess = tf.Session(config=session_conf)

    nn = TextHAN(sequence_length=x_train.shape[1],
                 num_classes=y_train.shape[1],
                 num_sentences = 3,
                 vocab_size=max_features,
                 embedding_size=embedding_size,
                 hidden_size=hidden_size,
                 batch_size=batch_size)

    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(nn.learning_rate)
    tvars = tf.trainable_variables()

    grads, _ = tf.clip_by_global_norm(tf.gradients(nn.loss, tvars), 5)
    grads_and_vars = tuple(zip(grads, tvars))
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    sess.run(tf.global_variables_initializer())
    batches = batch_iter(np.hstack((x_train,y_train)), batch_size, num_epochs)
    decay_speed = decay_coefficient*len(y_train)/batch_size
    counter = 0
    for batch in batches:
        learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-counter/decay_speed)
        counter += 1
        x_batch,y_batch = batch[:,:-2],batch[:,-2:]


        feed_dict = {nn.input_x: x_batch,
                     nn.input_y: y_batch,
                     nn.dropout_keep_prob: dropout_keep_prob,
                     nn.learning_rate: learning_rate}
        _, step, loss, accuracy= sess.run(
            [train_op, global_step, nn.loss, nn.accuracy],
            feed_dict)
        current_step = tf.train.global_step(sess, global_step)

        if current_step % evaluate_every == 0:
            print("\nEvaluation:")
            loss_sum = 0
            accuracy_sum = 0
            step = None
            batches_in_dev = len(y_dev) // batch_size
            for batch in range(batches_in_dev):
                start_index = batch * batch_size
                end_index = (batch + 1) * batch_size
                feed_dict = {
                        nn.input_x: x_dev[start_index:end_index],
                        nn.input_y: y_dev[start_index:end_index],
                        nn.dropout_keep_prob: 1.0
                    }
                step, loss, accuracy = sess.run(
                    [global_step, nn.loss, nn.accuracy],feed_dict)
                loss_sum += loss
                accuracy_sum += accuracy
            loss = loss_sum / batches_in_dev
            accuracy = accuracy_sum / batches_in_dev
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            print("")

    # predict test set
    all_predictions = []
    test_batches = batch_iter(x_test, batch_size, num_epochs=1, shuffle=False)
    for batch in test_batches:
        feed_dict = {
            nn.input_x: batch,
            nn.dropout_keep_prob: 1.0
        }
        predictions = sess.run([nn.predictions],feed_dict)[0]
        all_predictions.extend(list(predictions))



Evaluation:
2019-12-01T03:52:38.204614: step 100, loss 0.48834, acc 0.771205


Evaluation:
2019-12-01T03:53:42.306954: step 200, loss 0.37933, acc 0.843052


Evaluation:
2019-12-01T03:54:46.393631: step 300, loss 0.401685, acc 0.848493



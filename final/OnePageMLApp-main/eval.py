#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os

from tensorflow.keras.preprocessing import text

import data_helper
import re


def sentence_to_code(sentence):
    words = re.sub(r'[^\w]', ' ', sentence).split(' ')
    _x_test = np.array([tf.keras.preprocessing.text.one_hot(w, 7490) for w in words[:62] if w != ''])
    _x_test = np.array(_x_test).reshape(-1)
    if len(_x_test) < 62:
        zero_pad = 62 - len(_x_test)
        _x_test = np.hstack((_x_test, np.zeros(zero_pad, dtype=np.int64)))
    return _x_test


def classify_joke(sentences):
    checkpoint_file = os.path.dirname(os.path.abspath(__file__)) + "/runs/1574883205/checkpoints"
    # Eval Parameters
    tf.compat.v1.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
    tf.compat.v1.flags.DEFINE_string("checkpoint_dir", checkpoint_file, "Checkpoint directory from training run")
    tf.compat.v1.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

    # Misc Parameters
    tf.compat.v1.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.compat.v1.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

    FLAGS = tf.compat.v1.flags.FLAGS

    x_test = np.array([sentence_to_code(x) for x in sentences])

    # Evaluation
    # ==================================================
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.compat.v1.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.compat.v1.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.compat.v1.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = data_helper.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []

            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])
    return {s: p for s, p in zip(sentences, all_predictions)}

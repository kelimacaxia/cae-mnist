import tensorflow as tf
from tensorflow.contrib import rnn

n_input = 28
n_steps = 28
n_hidden = 128
n_classes = 10


def get_weight(shape):
    return tf.get_variable("weight", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))


def get_bias(shape):
    return tf.get_variable("bias", shape, initializer=tf.constant_initializer(0.0))


def RNN(x):
    x = tf.reshape(x, shape=[-1, n_steps, n_input])
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(axis=0, num_or_size_splits=n_steps, value=x)

    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    _weight = get_weight([n_hidden, n_classes])
    _bias = get_bias([n_classes])
    return tf.matmul(outputs[-1], _weight) + _bias

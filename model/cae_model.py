import tensorflow as tf

# WEIGHT AND BIASES
n1 = 64
n2 = 32
n3 = 1


def get_weight(shape):
    return tf.get_variable("weight", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))


def get_bias(shape):
    return tf.get_variable("bias", shape, initializer=tf.constant_initializer(0.0))


def cnn(_X):
    _input_r = tf.reshape(_X, shape=[-1, 28, 28, 1])
    # Encoder
    with tf.variable_scope("ce1"):
        _ce1_weight = get_weight([9, 9, 1, n1])
        _ce1_bias = get_bias([n1])
        _ce1 = tf.nn.sigmoid(tf.add(tf.nn.conv2d(_input_r, _ce1_weight, strides=[1, 1, 1, 1], padding='SAME'), _ce1_bias))

    with tf.variable_scope("ce2"):
        _ce2_weight = get_weight([1, 1, n1, n2])
        _ce2_bias = get_bias([n2])
        _ce2 = tf.nn.relu(tf.add(tf.nn.conv2d(_ce1, _ce2_weight, strides=[1, 1, 1, 1], padding='SAME'), _ce2_bias))

    with tf.variable_scope("ce3"):
        _ce3_weight = get_weight([5, 5, n2, n3])
        _ce3_bias = get_bias([n3])
        _ce3 = tf.nn.sigmoid(tf.add(tf.nn.conv2d(_ce2, _ce3_weight, strides=[1, 1, 1, 1], padding='SAME'), _ce3_bias))

    return _ce3

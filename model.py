import tensorflow as tf

# WEIGHT AND BIASES
n1 = 16
n2 = 32
n3 = 64
ksize = 5

def get_weight(shape):
    return tf.get_variable("weight", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))


def get_bias(shape):
    return tf.get_variable("bias", shape, initializer=tf.constant_initializer(0.0))


def cnn(_X, _keepprob):
    _input_r = tf.reshape(_X, shape=[-1, 28, 28, 1])
    # Encoder
    with tf.variable_scope("ce1"):
        _ce1_weight = get_weight([ksize, ksize, 1, n1])
        _ce1_bias = get_bias([n1])
        _ce1 = tf.nn.conv2d(_input_r, _ce1_weight, strides=[1, 2, 2, 1], padding='SAME')
        _ce1 = tf.nn.sigmoid(tf.add(_ce1, _ce1_bias))
        # _ce1 = tf.nn.dropout(_ce1, _keepprob)

    with tf.variable_scope("ce2"):
        _ce2_weight = get_weight([ksize, ksize, n1, n2])
        _ce2_bias = get_bias([n2])
        _ce2  = tf.nn.conv2d(_ce1, _ce2_weight, strides=[1, 2, 2, 1], padding='SAME')
        _ce2 = tf.nn.sigmoid(tf.add(_ce2, _ce2_bias))
        # _ce2 = tf.nn.dropout(_ce2, _keepprob)

    with tf.variable_scope("ce3"):
        _ce3_weight = get_weight([ksize, ksize, n2, n3])
        _ce3_bias = get_bias([n3])
        _ce3 = tf.nn.conv2d(_ce2, _ce3_weight, strides=[1, 2, 2, 1], padding='SAME')
        _ce3 = tf.nn.sigmoid(tf.add(_ce3, _ce3_bias))
        # _ce3 = tf.nn.dropout(_ce3, _keepprob)

    # Decoder
    # print(tf.shape(_input_r))
    with tf.variable_scope("cd3"):
        _cd3_weight = get_weight([ksize, ksize, n2, n3])
        _cd3_output_shape = [tf.shape(_X)[0], 7, 7, n2]
        _cd3_bias = get_bias([n2])
        _cd3 = tf.nn.conv2d_transpose(_ce3, _cd3_weight, _cd3_output_shape, strides=[1, 2, 2, 1], padding='SAME')
        _cd3 = tf.nn.sigmoid(tf.add(_cd3, _cd3_bias))
        # _cd3 = tf.nn.dropout(_cd3, _keepprob)

    with tf.variable_scope("cd2"):
        _cd2_weight = get_weight([ksize, ksize, n1, n2])
        _cd2_output_shape = [tf.shape(_X)[0], 14, 14, n1]
        _cd2_bias = get_bias([n1])
        _cd2 = tf.nn.conv2d_transpose(_cd3, _cd2_weight, _cd2_output_shape, strides=[1, 2, 2, 1], padding='SAME')
        _cd2 = tf.nn.sigmoid(tf.add(_cd2,_cd2_bias))
        # _cd2 = tf.nn.dropout(_cd2, _keepprob)

    with tf.variable_scope("cd1"):
        _cd1_weight = get_weight([ksize, ksize, 1, n1])
        _cd1_bias = get_bias([1])
        _cd1_output_shape = [tf.shape(_X)[0], 28, 28, 1]
        _cd1 = tf.nn.conv2d_transpose(_cd2, _cd1_weight, _cd1_output_shape, strides=[1, 2, 2, 1], padding='SAME')
        _cd1 = tf.nn.sigmoid(tf.add(_cd1, _cd1_bias))
        #_out = tf.nn.dropout(_cd1, _keepprob, name='output')

    return _cd1

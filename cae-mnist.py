import matplotlib.pyplot as plt
import numpy as np
# import math
# import random

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

from tensorflow.python.framework import ops
from tensorflow.python.framework import graph_util

from argparse import ArgumentParser

import model as model
import warnings
import os

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ops.reset_default_graph()

mnist = input_data.read_data_sets("data/", one_hot=True)


def train(options):
    batch_size = options.batch_size
    n_epochs = options.n_epochs

    trainimgs = mnist.train.images

    print("Network ready")
    learning_rate = 0.001

    dim = trainimgs.shape[1]
    x = tf.placeholder(tf.float32, [None, dim], name='x')
    # x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x')
    y = tf.placeholder(tf.float32, [None, dim])

    pred = model.cnn(x)  # ['out']
    cost = tf.reduce_sum(tf.square(pred - tf.reshape(y, shape=[-1, 28, 28, 1])), name='cost')
    optm = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    out_img = tf.reshape(pred, [28*28], name="out_img")

    print("Start training..")

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        print(pred)
        sess.run(init)
        for epoch_i in range(n_epochs):
            for batch_i in range(mnist.train.num_examples // batch_size):
                batch_xs, _ = mnist.train.next_batch(batch_size)
                trainbatch_noisy = batch_xs + 0.3 * np.random.randn(batch_xs.shape[0], 784)
                _, _cost = sess.run([optm, cost], feed_dict={x: trainbatch_noisy, y: batch_xs})
                print("[%02d/%02d] cost: %.4f" % (epoch_i, n_epochs, _cost))

        print(pred)

        print("Save model")
        graph_def = tf.get_default_graph().as_graph_def()
        output_graph = graph_util.convert_variables_to_constants(sess, graph_def, ['out_img'])
        with tf.gfile.GFile(os.path.join('./', 'mnist' + '.pb'), 'wb') as f:
            f.write(output_graph.SerializeToString())


def test():
    testimgs = mnist.test.images

    with tf.gfile.GFile(os.path.join('./', 'mnist' + '.pb'), "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        sess.run(init)
        x_ = sess.graph.get_tensor_by_name('x:0')
        out_ = sess.graph.get_tensor_by_name('out_img:0')

        test_xs, _ = mnist.test.next_batch(1)
        test_xs_noisy = test_xs + 0.3 * np.random.randn(test_xs.shape[0], 784)
        recon = sess.run(out_, feed_dict={x_: test_xs_noisy})
        fig, axs = plt.subplots(3, 1, figsize=(15, 4))

        axs[0].matshow(np.reshape(test_xs, (28, 28)), cmap=plt.get_cmap('gray'))
        axs[1].matshow(np.reshape(test_xs_noisy, (28, 28)), cmap=plt.get_cmap('gray'))
        axs[2].matshow(np.reshape(np.reshape(recon, (784,)), (28, 28)), cmap=plt.get_cmap('gray'))
        plt.show()


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--cmd', type=str, dest='cmd', default='train', help='command')
    parser.add_argument('--n_epochs', type=int, dest='n_epochs', default=3,
                        help='training n_epochs')
    parser.add_argument('--batch_size', type=int, dest='batch_size', default=128,
                        help='batch size')
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()
    print(options.cmd)
    if options.cmd == 'train':
        train(options)
    elif options.cmd == 'test':
        test()


if __name__ == '__main__':
    main()
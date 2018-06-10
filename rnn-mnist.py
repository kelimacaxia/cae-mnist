from tensorflow.python.framework import ops
from tensorflow.python.framework import graph_util

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

from argparse import ArgumentParser
import warnings
import os

from model import rnn_model as model

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ops.reset_default_graph()

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

learning_rate = 0.001
training_iters = 100000
display_step = 10

n_input = 28
n_steps = 28


def train(options):
    batch_size = options.batch_size

    dim = mnist.train.images.shape[1]
    x = tf.placeholder("float", [None, dim], name ='x')
    y = tf.placeholder("float", [None, 10], name='y')

    pred = model.RNN(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    output = tf.nn.softmax(pred, axis=1, name='output')

    correct_pred = tf. equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        step = 1
        while step * batch_size < training_iters:
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            if step % display_step == 0:
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
            step += 1
        print("Optimization Finished!")

        print("Save model")
        graph_def = tf.get_default_graph().as_graph_def()
        output_graph = graph_util.convert_variables_to_constants(sess, graph_def, ['output'])
        with tf.gfile.GFile(os.path.join('./', 'rnn-mnist' + '.pb'), 'wb') as f:
            f.write(output_graph.SerializeToString())


def test():
    with tf.gfile.GFile(os.path.join('./', 'rnn-mnist' + '.pb'), "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        sess.run(init)

        x_ = sess.graph.get_tensor_by_name('x:0')
        output_ = sess.graph.get_tensor_by_name('output:0')

        test_len = 1
        test_data, test_label = mnist.test.next_batch(test_len)
        print("Testing Accuracy:", sess.run(output_, feed_dict={x_: test_data}))


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
#!/usr/bin/env python3
# -*- coding : utf-8 -*-

__author__ = 'xuchao'
'a simple tensorflow'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# define a  add layer
def add_layer(inputs, in_size, out_size, activation_fuction=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    bias = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_p_bias = tf.matmul(inputs, Weights) + bias
    if activation_fuction is None:
        outputs = Wx_p_bias
    else:
        outputs = activation_fuction(Wx_p_bias)
    return outputs

# 1 prepare data
mnist = input_data.read_data_sets('MNIST', one_hot=True)

xs = tf.placeholder(dtype=tf.float32, shape=[None, 784])
ys = tf.placeholder(dtype=tf.float32, shape=[None, 10])

# calculate loss
predictions = add_layer(xs, 784, 10, activation_fuction=tf.nn.softmax)
loss = tf.reduce_mean(- tf.reduce_sum(ys * tf.log(predictions), reduction_indices=[1]))   # calculte the entropy loss

# train data
train = tf.train.GradientDescentOptimizer(0.4).minimize(loss)

# start train data
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # init all the variables
    for i in range(1000):
        batch_x, batch_y = mnist.train.next_batch(100)
        sess.run(train, feed_dict={xs: batch_x, ys: batch_y})
        if i % 50 == 0:
            pre_y = sess.run(predictions, feed_dict={xs: mnist.test.images})
            accuracy = tf.equal(tf.arg_max(pre_y, 1), tf.arg_max(mnist.test.labels, 1))
            accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))
            print("accuracy={}%".format(100 * sess.run(accuracy, feed_dict={xs: mnist.test.images, ys: mnist.test.labels})))
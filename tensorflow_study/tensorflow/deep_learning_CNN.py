#!/usr/bin/env python3
# -*- coding: UTF-8  -*_
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def compute_accuracy(v_xs, v_ys):
    global prediction   # 如果想改变这个变量的话,我们需要在局部函数中使用global关键字让他变成了全局变量
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})  # 另外注意一点,我们只是在训练集上面使用dropout,测试集合上面我们不在使用dropout方法
    correction = tf.equal(tf.arg_max(y_pre, 1), tf.arg_max(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correction, dtype=tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)  # tf.truncated分布比tf.normal分布要好?
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 这里的ksizes是pool层模板的大小,strides是每次模板移动的步数


# 1 load_data
mnist = input_data.read_data_sets('MNIST', one_hot='True')
xs = tf.placeholder(dtype=tf.float32, shape=[None, 784])
ys = tf.placeholder(dtype=tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(dtype=tf.float32)  # dropout radio

x_images = tf.reshape(xs, [-1, 28, 28, 1])  # 因为接下来需要输入到CNN当中,所以我们需要重新输入图像
# CNN_COV1
w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_images, w_conv1) + b_conv1)  # output: [None, 28, 28, 32], here,we usr relu to activate
h_pool1 = max_pool_2x2(h_conv1)  # output: [None, 14, 14, 32]  pool
# CNN_COV2
w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)  # output : [None, 14, 14, 64]
h_pool2 = max_pool_2x2(h_conv2)  # output: [None, 7, 7, 64]
# FC1
w_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])   # 将输出的结果进行reshape,注意reshape是变成-1而不是None,-1是可以自己计算大小的
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)   # output: [None, 1024]
h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob=keep_prob)    # here we use dropout to deviate the overfitting
# FC2
w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_dropout, w_fc2) + b_fc2)  # 在最后一层我们就不需要使用dropout策略了

# 2 train the network
entropy_loss = tf.reduce_mean(- tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))  # calculate the entropy loss
train = tf.train.AdamOptimizer(1e-4).minimize(entropy_loss)

# start train step
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        batch_x, batch_y = mnist.train.next_batch(100)
        sess.run(train, feed_dict={xs: batch_x, ys: batch_y, keep_prob: 0.5})
        if i % 50 == 0:  # 如果我们prediction不使用global的话,那么我们CNN模型就不能使用训练好的参数进行训练了,也就失去了意义
            print(compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000]))  # 测试1000张图片



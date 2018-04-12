#!/usr/bin/env python3
#! -*- coding: utf-8

__author__ = 'xuchao'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 训练集图像(55000 * 784),标签(55000 * 10) ; 测试集合(10000 * 784),标签(10000 * 10)

# 为了避免网络训练的时候出现全０梯度,这里首先对权重先进行简单处理


def weight_variable(shape):
    init = tf.truncated_normal(shape, stddev=0.1)  # 对权重施加一个简单的正数,防止出现全０梯度
    return tf.Variable(init)


def bias_variable(shape):
    init = tf.constant(0.1, shape=shape)  # 对偏差施加一个小小的正数
    return tf.Variable(init)
# 实现二层FC网络实现MNIST数据分类

# 1 读入数据


x_data = tf.placeholder(dtype=tf.float32, shape=[None, 784])  # 输入数据
y_data = tf.placeholder(dtype=tf.float32, shape=[None, 10])

FC1_w = weight_variable([784, 1024])   # 第一个全连接层
FC1_b = bias_variable([1024])
FC1 = tf.nn.relu(tf.matmul(x_data, FC1_w) + FC1_b)  # 使用relu作为激活函数

FC2_w = weight_variable([1024, 10])  # 第二个全连接层
FC2_b = bias_variable([10])
y_p = tf.nn.softmax(tf.matmul(FC1, FC2_w) + FC2_b)  # 使用softmax进行分类,预测

# 2 进行模型的训练

sess = tf.Session()
cross_entropy = - tf.reduce_sum(y_data * tf.log(y_p))  # 计算损失
train = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)  # 使用adam算法
predict = tf.equal(tf.argmax(y_data, 1),  tf.argmax(y_p, 1))  # 分别计算对应位置值是否一样
accuracy = tf.reduce_mean(tf.cast(predict, dtype=tf.float32))  # 计算平均准确度
sess.run(tf.global_variables_initializer())  # 全局变量的初始化.在tensorflow 里面所有的变脸需要进行初始化,如果这句话放在前面的话可能导致后面的没有进行初始化
for i in range(5000):

    batch_x, batch_y = mnist.train.next_batch(100)
    sess.run(train, feed_dict={x_data: batch_x, y_data: batch_y})
    if (i+1) % 200 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x_data:mnist.train.images, y_data:mnist.train.labels})
        print("train step", i+1, train_accuracy)
    if (i+1) % 500 == 0:
        test_accuracy = sess.run(accuracy, feed_dict={x_data:mnist.test.images, y_data:mnist.test.labels})
        print("test step", i+1, test_accuracy)


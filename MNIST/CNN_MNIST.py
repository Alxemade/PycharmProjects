#!/usr/bin/env python3
#! -*- coding: utf-8

__author__ = 'xuchao'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 训练集图像(55000 * 784),标签(55000 * 10) ; 测试集合(10000 * 784),标签(10000 * 10)

# 为了避免网络训练的时候出现全０梯度,这里首先对权重先进行简单处理
x_data = tf.placeholder(dtype=tf.float32, shape=[None, 784])  # 输入数据
y_data = tf.placeholder(dtype=tf.float32, shape=[None, 10])


def weight_variable(shape):
    init = tf.truncated_normal(shape, stddev=0.1)  # 对权重施加一个简单的正数,防止出现全０梯度
    return tf.Variable(init)


def bias_variable(shape):
    init = tf.constant(0.1, shape=shape)  # 对偏差施加一个小小的正数
    return tf.Variable(init)

# 定义卷积和池化


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')   # 这里的x就是输入图像,需要变成四维的tensor, W就是滤波器,stride是滑动的步长,padding表示是否进行0填充,'SAME'表示经过卷积层图像大小不发生变化

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 第一层卷积
W_conv1 = weight_variable([5, 5, 1, 32])  # 前面二个5是patch特征,1表示的输入的通道数,32是输出的通道数
b_conv1 = bias_variable([32])

# 为了应用这一层,我们把x变成一个4d向量,2,3维对应这图片的长宽,最后一维代表这颜色的通道数,灰度图像是1,rgb是3
x_image = tf.reshape(x_data, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+b_conv1)  # 通过上面的分析我们可以得到经过卷积层图像大小不发生变化, 仍然是28 * 28
h_pool1 = max_pool_2x2(h_conv1)  # 经过max_pool之后图像的大小变成14 * 14

# 第二层网络, 为了构建一个更深的网络,我们会把几个类似的层堆叠起来,第二层中,每个5*5的patch会得到64个特征
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 密集连接层, 现在图片大小是7*7, 我们需要加入一个1024个网络的全连接层.我们把池化层输出的张量reshape成一些向量,然后乘以权重矩阵,加上偏置,使用relu
w_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1)+b_fc1)

# Dropout层,为了减少过拟合;我们可以在训练的过程中启用dropout,而在测试的过程中关闭dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层,我们添加一个softmax层
w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2)+b_fc2)

sess = tf.Session()
cross_entropy = -tf.reduce_sum(y_data * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_data, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.global_variables_initializer())

for i in range(2000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = sess.run(accuracy, feed_dict={x_data: batch[0], y_data: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g" % (i, train_accuracy))

  sess.run(train_step, feed_dict={x_data: batch[0], y_data: batch[1], keep_prob: 0.5})
print("test accuracy %g" % sess.run(accuracy, feed_dict={x_data: mnist.test.images, y_data: mnist.test.labels, keep_prob: 1.0}))
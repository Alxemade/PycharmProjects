#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "xuchao"
'MNIST'

# 1 用tensorflow 导入数据
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)   # 导入一个Database了

# print('training data shape ', mnist.train.images.shape) 训练集的图像是55000 * 784
# print(mnist.train.labels.shape)  训练集的标签是55000 * 10,因为这里采用ont-hot 编码
# print(mnist.test.images.shape)  测试集图像是10000 * 784
# print(mnist.test.labels.shape)   测试图像的标签是10000 * 10

# 2 构建模型, 一般输入输出数据使用placehold,而训练的数据使用variable
x_data = tf.placeholder(dtype=tf.float32, shape=[None, 784])
y_data = tf.placeholder(dtype=tf.float32, shape=[None, 10])

w = tf.Variable(tf.zeros([784, 10]))  # 因为一开始不知道模型参数,所以我们使用0,这里需要注意一点,zeros里面需要打上一个[]
b = tf.Variable(tf.zeros([10]))


y_p = tf.nn.softmax(tf.matmul(x_data, w) + b)  # 构建softmax网络进行预测

h_ = - tf.reduce_sum(tf.multiply(y_data, y_p))  # 计算出交叉熵

sess = tf.Session()
sess.run(tf.global_variables_initializer())

loss = tf.train.GradientDescentOptimizer(0.01).minimize(h_)  # 使用梯度下降法进行计算损失

for _ in range(1000):
    batch_x, batch_y = mnist.train.next_batch(100)  # 每次从数据中随机选择100个数据
    sess.run(loss, feed_dict={x_data: batch_x, y_data: batch_y})  # 已经训练好参数了

# 计算准确度
predict = tf.equal(tf.argmax(y_data, 1), tf.argmax(y_p, 1))  # 比较对应位置值是否一致
accruacy = tf.reduce_mean(tf.cast(predict, dtype=tf.float32))   # 计算准确度
print(sess.run(accruacy, feed_dict={x_data: mnist.test.images, y_data: mnist.test.labels}))  # 得到的参数在验证集上



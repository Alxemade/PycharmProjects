#!/usr/bin/env python3
# -*- coding : utf-8 -*-

__author__ = 'xuchao'
'a simple tensorflow'

import tensorflow as tf
from sklearn.datasets import load_digits   # use sklearn to import data
from sklearn.preprocessing import LabelBinarizer  # 将普通的数据变成了one_hot编码
from sklearn.model_selection import train_test_split   # 将数据分成train和test集合
import numpy as np

# define a  add layer
def add_layer(inputs, in_size, out_size, layer_name, activation_fuction=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    bias = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_p_bias = tf.matmul(inputs, Weights) + bias
    Wx_p_bias = tf.nn.dropout(Wx_p_bias, keep_prob)  # 这里的keep_prob难道是全局变量?
    if activation_fuction is None:
        outputs = Wx_p_bias
    else:
        outputs = activation_fuction(Wx_p_bias)
    tf.summary.histogram(layer_name + '\outputs', outputs)  # use the tensorboard to show the outputs
    return outputs

# 1 prepare data
digits = load_digits()
X = digits.data  # 1797 * 64
y = digits.target  # 这个是普通的数据0-9
y = LabelBinarizer().fit_transform(y)  #  使用这种手法我们就可以将其变成了one-hot 编码了
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)  # 按照测试集占了整个集合的0.33来进行切分

keep_prob = tf.placeholder(dtype=tf.float32)  # 这个是控制着dropout的比重的
xs = tf.placeholder(dtype=tf.float32, shape=[None, 64])
ys = tf.placeholder(dtype=tf.float32, shape=[None, 10])

# calculate loss
l1 = add_layer(xs, 64, 50, 'l1', activation_fuction=tf.nn.tanh)
predictions = add_layer(l1, 50, 10, 'l2', activation_fuction=tf.nn.softmax)
loss = tf.reduce_mean(- tf.reduce_sum(ys * tf.log(predictions), reduction_indices=[1]))   # calculte the entropy loss
tf.summary.scalar('loss', loss)
# train data
train = tf.train.GradientDescentOptimizer(0.4).minimize(loss)
sess = tf.Session()
merged = tf.summary.merge_all()  # 先把所有的summary集合起来
train_write = tf.summary.FileWriter('logs/train', sess.graph)   # 写入logs file
test_write = tf.summary.FileWriter('logs/test', sess.graph)

sess.run(tf.global_variables_initializer())  # init all the variables
for i in range(1000):
    sess.run(train, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
    if i % 50 == 0:
        train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})  # 也是需要运行所有的summaries才是真正的运行程序
        test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})  # 也是需要
        train_write.add_summary(train_result, i)   # file into the file
        test_write.add_summary(test_result, i)

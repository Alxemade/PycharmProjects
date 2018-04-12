#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "xuchao"
'activation test'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 200)  # 产生一个横坐标
y_relu = tf.nn.relu(x)  # 产生一个relu的激活函数
y_sigmoid = tf.nn.sigmoid(x)  # 产生一个sigmodi函数
y_tanh = tf.nn.tanh(x)  # 产生一个tanh激活函数
y_sotfplus = tf.nn.softplus(x)  # 产生一个softplus激活函数

with tf.Session() as sess:
    y_relu, y_sigmoid, y_tanh, y_sotfplus = sess.run([y_relu, y_sigmoid, y_tanh, y_sotfplus])  # 这里注意一点,当需要同时输出多个变量的时候,需要将其变成一个list也就是一个tensor

plt.figure(1, figsize=(8, 6))

plt.subplot(2, 2, 1)
plt.plot(x, y_relu, c='red', label='relu')  # 注意需要之前运行sess.run如果直接给他显示会出现错误的
plt.legend(loc='best')  # 这个是设置图例
plt.ylim(-1, 5)  # 这个是设置y的坐标范围

plt.subplot(2, 2, 2)
plt.plot(x, y_sigmoid, c='red', label='sigmoid')
plt.legend(loc='best')
plt.ylim(-0.2, 1.2)

plt.subplot(2, 2, 3)
plt.plot(x, y_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')

plt.subplot(2, 2, 4)
plt.plot(x, y_sotfplus, c='red', label='softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')
plt.show()

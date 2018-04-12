#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'xuchao'
'tensorflow test'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#  inputs : input the data
#  in_size : input the weight size
#  out_size: output the weight size
#  activation_function: if or not use the activation function


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    bias = tf.Variable(tf.zeros([1, out_size]))
    result = tf.matmul(inputs, Weights) + bias
    if activation_function is None:
        outputs = result
    else:
        outputs = activation_function(result)
    return outputs

# 1 first we need to prepare some data
x_data = np.linspace(-1, 1, 1000, dtype=np.float32)[:, np.newaxis]  # 这里我们为什么需要扩展维度？因为我们需要讲数据变成2维才可以使用
noise = np.random.normal(0, 0.05, size=x_data.shape).astype(np.float32)  # 因为tensorflow默认都是float32,而这里产生的数据是float64的,所以我们需要进行强制类型转换
y_data = np.square(x_data) - 0.5 + noise

fig, ax = plt.subplots()
ax.scatter(x_data, y_data)  # 默认我们只能显示一次数据，如果想要动态的显示全部的过程,我们需要使用
plt.ion()  # 进入交互模式,可以一直产生图像，要不然程序仅仅会显示第一次
plt.show()
xs = tf.placeholder(dtype=tf.float32, shape=[None, 1])   # 这里我们将一开始数据变成(300,1)这里我们就可以喂数据了
ys = tf.placeholder(dtype=tf.float32, shape=[None, 1])

# add layer
hid1 = add_layer(x_data, 1, 10, activation_function=tf.nn.relu)  # 隐藏层1,卷积尺寸是(1*10)使用relu的激活函数
predictions = add_layer(hid1, 10, 1, activation_function=None)  # 输出层，不使用激活函数
# 计算损失
loss = tf.reduce_mean(tf.square(y_data - predictions))

train = tf.train.AdadeltaOptimizer(0.1).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # 全局变量初始化
    for _ in range(10000):
        sess.run(train, feed_dict={xs: x_data, ys: y_data})
        if _ % 200 == 0:
            try:
                ax.lines.remove(lines[0])   # 这个是可以删除之前的数据的

            except Exception:
                pass
            #print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
            pre_data = sess.run(predictions, feed_dict={xs: x_data})
            lines = ax.plot(x_data, pre_data, 'r-', lw=5)
            #ax.lines
            plt.pause(0.5)  # 暂停0.5s,可以知道逼近的整个过程，要不暂停程序运行速度过快，看不到过程了



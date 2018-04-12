#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author = "xuchao"

import  tensorflow as tf
import numpy as np

# 1 prepare data
x_data = np.random.rand(100)  # 默认是float64,随机产生100个数,而且产生的数都是正数
y_data = x_data * 0.1 + 0.3

# 2 构建模型
w = tf.Variable(tf.random_uniform([1], -1, 1))  # 这里由于是直线拟合任务,仅仅需要一个参数,这里uniform接受的是秩为1,需要打个跑[]
b = tf.zeros([1])  # 由于版本更新了所以对于一维tensor来说,加不加[]都是可以的
y_p = w * x_data + b
loss = tf.reduce_mean(tf.square(y_data - y_p))  # 计算损失函数

# 3 调用优化器进行优化
opt = tf.train.GradientDescentOptimizer(0.01)
train = opt.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)  # 对于变量来说需要运行初始化任务
    for i in range(100):
        sess.run(train)
        if i % 10 == 0:
            print("i={},loss={}".format(i, sess.run(loss)))
    print("w={},b={}".format(sess.run(w),sess.run(b)))
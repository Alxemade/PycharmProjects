#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "xuchao"
'test session'

import tensorflow as tf

a = tf.constant([[2, 2]])
b = tf.constant([[3], [3]])
c = tf.matmul(a, b)  # 输出两个数的矩阵乘法
d = tf.multiply(a, b)  # 这个是矩阵的向量积

# 1 . 第一种写法,需要自己关闭close
sess = tf.Session()
print(sess.run(c))
sess.close()

# 2. 第二种写法,不需要自己close
with tf.Session() as sess:
    print(sess.run(c))  # 这种写法不需要自己关闭close
    
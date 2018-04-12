#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "xuchao"

import tensorflow as tf

a = tf.constant(2.0)
b = tf.constant(3.0)
c = tf.constant(5.0)

d1 = tf.add(a, b)
d2 = tf.multiply(b, c)
with tf.Session() as sess:
    print(sess.run([d1, d2]))  # 输出[5.0, 15.0]
    # 直接写sess.run(d1, d2)是不对的出现 raise TypeError("Using a `tf.Tensor` as a Python `bool` is not allowed.
    print(sess.run([d1, d2])[0])  # 如果想要输出第一个数可以这么写

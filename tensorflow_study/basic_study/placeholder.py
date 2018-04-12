#!/usr/bin/env python3
# -*- coding: utf-8

__author__ = "xuchao"
'placeholder test'

import tensorflow as tf

a = tf.placeholder(dtype=tf.float32)
b = tf.placeholder(dtype=tf.float32)

c = tf.multiply(a, b)
with tf.Session() as sess:
    print(sess.run(c, feed_dict={a: [1.], b: [2.]}))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "xuchao"

import tensorflow as tf

state = tf.Variable(0)   # 这是创建的初始值
one = tf.constant(1)   # 这是作为加1操作
new_value = tf.add(state, one)  # 这是作为新的状态
update = tf.assign(state, new_value)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(state))

    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
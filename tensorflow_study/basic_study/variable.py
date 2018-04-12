#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "xuchao"
'variable test'

import tensorflow as tf

state = tf.Variable(0, name="counter")
one = tf.constant(1)
new_state = tf.add(state, one)
update = tf.assign(state, new_state)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(4):
        sess.run(update)
        print(sess.run(state))
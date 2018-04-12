#/usr/bin/env python3
#! -*- coding: utf-8 -*-

'variables'
__author__ = 'xuchao'

import tensorflow as tf

# create a variable with a random value
weight = tf.Variable(tf.random_normal([784, 200], name="weights"))

# create another variable with the same value
w2 = tf.Variable(weight.initial_value(), names="w2")  # 使用另一个变量进行赋值

sess = tf.Session()
print(sess.run("weight", weight))
print(sess.run("w2", w2))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 对几个数据求平均
__author__ = "xuchao"

import tensorflow as tf

sumdata = tf.Variable(0.0, dtype=tf.float32)  # 作为存储的数据
vec = tf.constant([1.0, 2.0, 3.0, 4.0])
add = tf.placeholder(tf.float32)
hew = tf.add(sumdata, add)
update = tf.assign(sumdata, hew)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(sumdata))
    print(sess.run(vec))

    for _ in range(4):
        sess.run(update, feed_dict={add: sess.run(vec[_])})
        print(sess.run(sumdata))
    print(sess.run(sumdata/4.0))  # 以前是不可以直接除以4的,现在可能进行修改了吧,是可以直接除以数字的,这样的话就比较方便了

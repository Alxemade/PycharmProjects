#!/usr/bin/env python3
#! -*- coding: utf-8 -*-

import tensorflow as tf

with tf.name_scope('graph') as scope:   # f.name_scope函数是作用域名?
    matrix1 = tf.constant([[3., 3.]], name="matrix1")
    matrix2 = tf.constant([[2.], [2.]], name="matrix2")
    product = tf.matmul(matrix1, matrix2, name="product")

sess = tf.Session()

writer = tf.summary.FileWriter("logs/", sess.graph)   # 这句话是什么意思?暂时先不管,先把他跑通

init = tf.global_variables_initializer()

sess.run(init)
sess.run(product)
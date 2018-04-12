#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "xuchao"
'MNIST'

# 1 用tensorflow 导入数据
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)   # 导入一个Database了

# print('training data shape ', mnist.train.images.shape) 训练集的图像是55000 * 784
# print(mnist.train.labels.shape)  训练集的标签是55000 * 10,因为这里采用ont-hot 编码
# print(mnist.test.images.shape)  测试集图像是10000 * 784
# print(mnist.test.labels.shape)   测试图像的标签是10000 * 10

# 构建一个多层的神经网络

# 1 权重初始化 这个模型中的权重在初始化的时候加入少量的噪声来打破对称性,避免0梯度,一个好的方法就是用一个较小的正数来初始化偏置

def weight_variabl(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 2 卷积和池化

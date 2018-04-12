#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'xuchao'
'tensorflow test'
import tensorflow as tf
import numpy as np

#  inputs : input the data
#  in_size : input the weight size
#  out_size: output the weight size
#  nlayer: visual which layer to show on the tensorboard
#  activation_function: if or not use the activation function


def add_layer(inputs, in_size, out_size, nlayer, activation_function=None):
    layer_name = 'layer % s' % nlayer  # 把一个字符串传给了layer_name，这样显示就可以显示那一层的参数了,这个思想还是相当不错的
    with tf.name_scope('layer'):   # 这个总体名字是要写在函数定义的之前的,这样我们得到的tensorboard会出现分级的显示
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]))
            tf.summary.histogram(layer_name + '/Weights', Weights)  # 显示weights参数
        with tf.name_scope('bias'):
            bias = tf.Variable(tf.zeros([1, out_size]))
            tf.summary.histogram(layer_name + '/bias', bias)
        with tf.name_scope('result'):
            result = tf.matmul(inputs, Weights) + bias
        if activation_function is None:
            outputs = result
        else:
            outputs = activation_function(result)
        tf.summary.histogram(layer_name + '/outputs', outputs)
        return outputs

# 1 first we need to prepare some data
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]  # 这里我们为什么需要扩展维度？因为我们需要讲数据变成2维才可以使用
noise = np.random.normal(0, 0.05, size=x_data.shape).astype(np.float32)  # 因为tensorflow默认都是float32,而这里产生的数据是float64的,所以我们需要进行强制类型转换
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('inputs'):
    xs = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='input1')   # 这里我们将一开始数据变成(300,1)这里我们就可以喂数据了
    ys = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='input2')

# add layer
hid1 = add_layer(xs, 1, 10, nlayer=1, activation_function=tf.nn.relu)  # 隐藏层1,卷积尺寸是(1*10)使用relu的激活函数
predictions = add_layer(hid1, 10, 1, nlayer=2, activation_function=None)  # 输出层，不使用激活函数
# 计算损失
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(ys - predictions))
    tf.summary.scalar('loss', loss)  # 显示损失是在event上面显示
with tf.name_scope('train'):
    train = tf.train.AdadeltaOptimizer(0.1).minimize(loss)

sess = tf.Session()
merged = tf.summary.merge_all()   # Merges all summaries collected in the default graph.
write = tf.summary.FileWriter('logs/', sess.graph)  # 这是最新的使用手法,写入本地log文件,注意log/这种表达方式
sess.run(tf.global_variables_initializer())  # 初始化所有的变量

for i in range(1000):
    sess.run(train, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        ref = sess.run(merged, feed_dict={xs: x_data, ys: y_data})  # 也是需要运行所有的summaries才是真正的运行程序
        write.add_summary(ref, i)
        # 前面执行write.add_summary是错误的,原因就是add_summary写入的形参需要是summary,
        # write.add_summary(merged,i)因为传入的数据merged是tensor,二者类型是不匹配的,所以类型出现了错误也是很正常的,

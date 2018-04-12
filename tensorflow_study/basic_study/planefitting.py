
#!/usr/bin/env python3
# -*- coding: utf-3 -*-

' a plane fitting '
__author__ = "xuchao"

# 1 prepare data 2 conduct the model 3 solve the model
import numpy as np
import tensorflow as tf

# 1 准备数据
x_data = np.float32(np.random.rand(2, 100))  # generate 100 point
# 这里还需要注意一点我们tensorflow里面的数据默认是float32的,所以这里的我们使用np.random.rand创建的数据是float64的.所以我们需要进行强制类型转换成float32
y_data = np.array([0.1, 0.2]) @ x_data + 0.3  # (1*2) @ (2*100) = (1*100)

# 2 构建模型
b = tf.Variable(tf.zeros([1]))  # 因为这是我们需要训练的参数所以是需要设置成变量的进行存储
w = tf.Variable(tf.random_uniform(shape=(1, 2), minval=-1, maxval=1))
y = tf.matmul(w, x_data) + b

init = tf.global_variables_initializer()  # 针对变量我们需要进行全局初始化操作
sess = tf.Session()
sess.run(init)  # 这才是运行variable操作

# 求解模型
loss = tf.reduce_mean(tf.square(y - y_data))  # 求解均方损失
optimizer = tf.train.GradientDescentOptimizer(0.5)  # 设置基于梯度下降法学习率是0.5
train = optimizer.minimize(loss)    # 最小化损失
# print(sess.run(loss))

for step in range(0, 501):  # 运行500次循环
    sess.run(train)  # 进行数据训练
    if step % 20 == 0:
        print("step", step, sess.run(w), sess.run(b))  # 每隔20次输出一次中间结果

sess.close()  # 当运行Session之后需要关闭session


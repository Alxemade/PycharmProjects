"""
visit https://morvanzhou.github.io/tutorials/ for more!

Build two networks.
1. Without batch normalization
2. With batch normalization

Run tests on these two networks.
"""

# 23 Batch Normalization

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


ACTIVATION = tf.nn.relu  # 设置全局的激活函数
N_LAYERS = 7  # 设置整个神经网络的层数
N_HIDDEN_UNITS = 30


def fix_seed(seed=1):  # 设置随机数种子？
    # reproducible
    np.random.seed(seed)
    tf.set_random_seed(seed)


def plot_his(inputs, inputs_norm):  # 这个不是我们关注的重点我们暂时先不看,函数主要的功能就是绘制各个层的状况吗
    # plot histogram for the inputs of every layer
    for j, all_inputs in enumerate([inputs, inputs_norm]):
        for i, input in enumerate(all_inputs):
            plt.subplot(2, len(all_inputs), j*len(all_inputs)+(i+1))
            plt.cla()
            if i == 0:
                the_range = (-7, 10)
            else:
                the_range = (-1, 1)
            plt.hist(input.ravel(), bins=15, range=the_range, color='#FF5733')
            plt.yticks(())
            if j == 1:
                plt.xticks(the_range)
            else:
                plt.xticks(())
            ax = plt.gca()
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
        plt.title("%s normalizing" % ("Without" if j == 0 else "With"))
    plt.draw()
    plt.pause(0.01)


def built_net(xs, ys, norm):  # norm参数表示是否进行BN算法
    def add_layer(inputs, in_size, out_size, activation_function=None, norm=False):  # 对中间层进行BN
        # weights and biases (bad initialization for this case)
        Weights = tf.Variable(tf.random_normal([in_size, out_size], mean=0., stddev=1.))
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)

        # fully connected product
        Wx_plus_b = tf.matmul(inputs, Weights) + biases

        # normalize fully connected product,这个BN算法执行的位置是在网络通过activation函数之前设置的的
        if norm:  # 表示如果此时需要进行bn算法
            # Batch Normalize  返回这一批数据的均值和方差备用
            fc_mean, fc_var = tf.nn.moments(
                Wx_plus_b,
                axes=[0],   # the dimension you wanna normalize, here [0] for batch
                            # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
            )  # 这里相当于fc_mean = ub , fc_var = 方差
            # 因为这里需要进行重构x, y = scale * x + shift 所以二者的大小应该和输出大小一样,out_size
            scale = tf.Variable(tf.ones([out_size]))   # 重构中尺度变换
            shift = tf.Variable(tf.zeros([out_size]))  # 重构中平移变换,这二个是为了增加重构的多样性,网络自己学习参数
            epsilon = 0.001  # 在进行批量归一化的时候防止除数为０的操作

            # apply moving average for mean and var when train on batch
            ema = tf.train.ExponentialMovingAverage(decay=0.5)  # 接下来使用指数滑动平均,在某些证明中对于参数进行平均可以提高网络性能
            def mean_var_with_update():  # 暂时先不管为什么定义函数？暂时是为了方便吗？后面使用到了m? -- 简化程序！
                ema_apply_op = ema.apply([fc_mean, fc_var])  # 输入参数是list,创建影子变量,并进行滑动平均
                with tf.control_dependencies([ema_apply_op]):  #　控制程序流,仅仅当上面创建影子变量并滑动平均下面语句才可以执行
                    return tf.identity(fc_mean), tf.identity(fc_var)  #　首先里面需要是一个op,然后其实返回的是更新之后影子变量的值,这个二句话相当分别接收上面的list
            mean, var = mean_var_with_update()  # 接收进行EMA之后的值　

            Wx_plus_b = tf.nn.batch_normalization(Wx_plus_b, mean, var, shift, scale, epsilon)  # 相当于进行归一化以及重构都结合了　
            # similar with this two steps:
            # Wx_plus_b = (Wx_plus_b - fc_mean) / tf.sqrt(fc_var + 0.001)
            # Wx_plus_b = Wx_plus_b * scale + shift　这里为什么需要进行重构了,因为经过上面的归一化之后,相当于是一个正态分布,所以 为了增加模型的稳健性,增加一个尺度和平移参数

        # activation
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)

        return outputs

    fix_seed(1)  # 这个暂时不管

    if norm:  # 对输入层也进行bn算法, 从模型复用的角度来看,这样分别对输入层和隐藏层进行BN算法,有点冗余.
        # BN for the first input
        fc_mean, fc_var = tf.nn.moments(
            xs,   # 这里是对输入数据进行BN算法
            axes=[0],
        )
        scale = tf.Variable(tf.ones([1]))
        shift = tf.Variable(tf.zeros([1]))
        epsilon = 0.001
        # apply moving average for mean and var when train on batch
        ema = tf.train.ExponentialMovingAverage(decay=0.5)
        def mean_var_with_update():
            ema_apply_op = ema.apply([fc_mean, fc_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(fc_mean), tf.identity(fc_var)
        mean, var = mean_var_with_update()
        xs = tf.nn.batch_normalization(xs, mean, var, shift, scale, epsilon)

    # record inputs for every layer
    layers_inputs = [xs]  # 这个是为了接下来的画图吗

    # build hidden layers
    for l_n in range(N_LAYERS):
        layer_input = layers_inputs[l_n]
        in_size = layers_inputs[l_n].get_shape()[1].value  # 这个得到的是每一层的输入数据

        output = add_layer(
            layer_input,    # input
            in_size,        # input size
            N_HIDDEN_UNITS, # output size
            ACTIVATION,     # activation function
            norm,           # normalize before activation
        )
        layers_inputs.append(output)    # add output for next run

    # build output layer
    prediction = add_layer(layers_inputs[-1], 30, 1, activation_function=None)  # 取上面中间层最后一层作为输出层的输入层
    # 上面的语句表示神经网络已经建立完毕
    cost = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
    return [train_op, cost, layers_inputs]  # 作为后面的画图方便吧

# make up data
fix_seed(1)
x_data = np.linspace(-7, 10, 2500)[:, np.newaxis]  # (2500　＊ 1 ) data
np.random.shuffle(x_data)  # 打乱数据,减少数据之间的相关性
noise = np.random.normal(0, 8, x_data.shape)  # 增加噪声
y_data = np.square(x_data) - 5 + noise  # 得到输出数据

# plot input data
plt.scatter(x_data, y_data)
plt.show()   # 显示数据

xs = tf.placeholder(tf.float32, [None, 1])  # [num_samples, num_features] 产生输入数据
ys = tf.placeholder(tf.float32, [None, 1])

train_op, cost, layers_inputs = built_net(xs, ys, norm=False)   # 产生没有BN的普通网络
train_op_norm, cost_norm, layers_inputs_norm = built_net(xs, ys, norm=True) # 产生存在BN的网络

sess = tf.Session()
init = tf.global_variables_initializer()  # 全局初始化
sess.run(init)

# record cost
cost_his = []  # 无BN
cost_his_norm = []  # 存在BN这个是为了显示每个层准备的
record_step = 5  # 这个是每隔多少次传入一次loss并且显示

plt.ion()  # 这个是让图表进入交互模式,可以画出动态图表
plt.figure(figsize=(7, 3))  # 设置图像显示
for i in range(250):
    if i % 50 == 0:   # 每隔50次绘制一次图像,这个时候网络已经学习好一部分的参数了,我们看看各个层的变化
        # plot histogram
        all_inputs, all_inputs_norm = sess.run([layers_inputs, layers_inputs_norm], feed_dict={xs: x_data, ys: y_data}) # here, we give all the data
        plot_his(all_inputs, all_inputs_norm)  # 画出各个图的变化,分别是加入没有BN,有BN区别
        # plt.pause(2)  # 没隔2s显示一下
    # train on batch
    sess.run([train_op, train_op_norm], feed_dict={xs: x_data[i*10:i*10+10], ys: y_data[i*10:i*10+10]})  # 这是训练的过程,所以我们并没有给所有的数据,而是一小批的数据,10个数据组成一批

    if i % record_step == 0:

        # record cost
        cost_his.append(sess.run(cost, feed_dict={xs: x_data, ys: y_data}))  # 传入不加BN的loss
        cost_his_norm.append(sess.run(cost_norm, feed_dict={xs: x_data, ys: y_data}))  # 传入加入BN的loss

plt.ioff()  # 关闭图表连续显示, 要不然一下子就闪过了
plt.figure()
plt.plot(np.arange(len(cost_his))*record_step, np.array(cost_his), label='no BN')     # 这里我们显示不使用bn的效果
plt.plot(np.arange(len(cost_his))*record_step, np.array(cost_his_norm), label='BN')   # 这里我们显示使用BN的效果
plt.legend()  # 显示图列
plt.show()


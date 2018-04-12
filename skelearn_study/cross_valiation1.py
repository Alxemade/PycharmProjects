#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'learning curve'
__author__ = 'xuchao'

from sklearn.datasets import load_digits   # load data
from sklearn.svm import SVC    #  use the svc to discriminate the digits datasets
from sklearn.model_selection import validation_curve  # learning the curve on the different size of data
import matplotlib.pyplot as plt
import numpy as np
# 1 prepare data
digits = load_digits()
X = digits.data
y = digits.target
# 2 train the mode
param_range = param_range = np.logspace(-6, -2.3, 5)
train_loss, test_loss = validation_curve(SVC(), X, y, param_name='gamma', param_range=param_range, cv=10, scoring='neg_mean_squared_error')
# 这里需要注意validation_curve中的参数param_name=需要是estimator里面的形参,而不是任意一个数, param_range是需要进行测试的参数列表
# 3 visual the figure
train_loss_mean = - np.mean(train_loss, axis=1)  # 这里的损失都是负的,为了显示的好处我们加上-号
test_loss_mean = - np.mean(test_loss, axis=1)   # 这里是二维数据,横的是%数据,列是第k折数据,所以我们需要对每一行求平均
# 4 visual the data
plt.figure()
plt.plot(param_range, train_loss_mean, 'o-', color="r", label="train loss") # 这里注意一下需要使用'o-'才可以显示点和线
plt.plot(param_range, test_loss_mean, 'o-', color="b", label="cross valiation")
plt.legend(loc='best')  # 显示图例
plt.xlabel('train sizes')
plt.ylabel('loss')
plt.show()

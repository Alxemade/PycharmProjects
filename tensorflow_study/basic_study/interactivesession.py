# 为了便于使用诸如 IPython 之类的 Python 交互环境, 可以使用 InteractiveSession 代替 Session 类,
# 使用 Tensor.eval() 和 Operation.run() 方法代替 Session.run(). 这样可以避免使用一个变量来持有会话.

import tensorflow as tf

'''a = tf.constant(1.0)
b = tf.constant(2.0)
c = a + b

# 下面的二种情况是等价的
with tf.Session() as sess:
    print(sess.run(c))  # 输出3.0
    print(c.eval())     # 输出3.0

sess = tf.InteractiveSession()
print(c.eval())
sess.close()
'''

a = tf.constant(1.0)
b = tf.constant(2.0)
c = tf.Variable(3.0)
d = a + b

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# print(a.run()) 'Tensor' object has no attribute 'run'
print(a.eval())   # Tensor是存在eval()方法的

# run()方法主要用来
x = tf.Variable(1.2)
# print(x.eval()) 变量还没有初始化,这时候暂时还不能调用
x.initializer.run()  # x.initializer是一个初始化op,只有op才可以调用run方法
print(x.eval())
sess.close()

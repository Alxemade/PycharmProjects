
import tensorflow as tf

# (1) 首先构建一个图

# 创建一个常量op.这个op称为一个节点
# 加到默认图中

matrix1 = tf.constant([[3., 3.]])

# 构建第二个op
matrix2 = tf.constant([[2.], [2.]])

# 创建一个矩阵的乘法
product = tf.matmul(matrix1, matrix2)

# (2)在一个回话中启动图, 使用Session对象,如果没有任何创建参数,启动默认图

# 启动默认图
'''
sess = tf.Session()

# 调用 sess 的 'run()' 方法来执行矩阵乘法 op, 传入 'product' 作为该方法的参数.
# 上面提到, 'product' 代表了矩阵乘法 op 的输出, 传入它是向方法表明, 我们希望取回
# 矩阵乘法 op 的输出.
#
# 整个执行过程是自动化的, 会话负责传递 op 所需的全部输入. op 通常是并发执行的.
#
# 函数调用 'run(product)' 触发了图中三个 op (两个常量 op 和一个矩阵乘法 op) 的执行.
#
# 返回值 'result' 是一个 numpy `ndarray` 对象.
result = sess.run(product)

# 打印输出
print(type(result))  # 打印输出类型
print(result)  # 打印输出值

# 任务完成,关闭会话
sess.close()
'''

# Session 对象在使用完后需要关闭以释放资源. 除了显式调用 close 外, 也可以使用 "with" 代码块 来自动完成关闭动作
with tf.Session() as sess:
    result = sess.run(product)
    print(result)


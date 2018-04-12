import tensorflow as tf
a = tf.constant([[1, 2], [3, 4]])
with tf.Session() as sess:
    #mean = sess.run(tf.reduce_mean(a)) 理论我们需要输出2.5但是通过这种方法,我们并不能输出正确的结果
    sum = tf.reduce_sum(a, reduction_indices=[1])   # 一种方法是通过reduction_indices使用进行降维,这个如果是0则按照多少列输出多少个结果,如果是1表示多少行输出多少个数据
    sum1 = tf.reduce_sum(a, 1)  # 一种方法是加上一个1表示列进行合并
    mean = tf.reduce_mean(tf.reduce_sum(a, reduction_indices=[1]))
    mean1 = tf.reduce_sum(a, [0, 1])  # 这个是对所有数据进行求和操作
    print(sess.run(mean1))
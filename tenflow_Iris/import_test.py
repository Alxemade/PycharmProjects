import tensorflow as tf

dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_normal([4,10,1]))
print(dataset1.output_types)
print(dataset1.output_shapes)  # 这个仅仅输出部分形状


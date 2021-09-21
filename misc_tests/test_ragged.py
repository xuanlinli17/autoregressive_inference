import tensorflow as tf

@tf.function
def f():
    a = tf.range(6)
    a = tf.reshape(a, [2,1,3])
    a = tf.RaggedTensor.from_tensor(a)
    tf.print(a)
    b = tf.range(12)
    b = tf.reshape(b, [2,1,6])
    b = tf.RaggedTensor.from_tensor(b)
    tf.print(b)
    tf.print(tf.concat([a,b], axis=1))

f()

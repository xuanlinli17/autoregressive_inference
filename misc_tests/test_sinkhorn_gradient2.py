from voi.nn.base.sinkhorn import *
import tensorflow as tf
import tensorflow_probability as tfp

if __name__ == "__main__":

    def sinkhorn_eager(x):
        for _ in range(200):
            x = tf.math.log_softmax(x, axis=-2)
            x = tf.math.log_softmax(x, axis=-1)
        return tf.math.exp(x)

    tf.random.set_seed(1) 
    n = 10
    x = tf.random.normal([1, n, n])
    skh = Sinkhorn(200)
    print("original x", x)

    with tf.GradientTape() as tape:
        tape.watch(x)
        soft_permu = skh.call(x)
        print("soft_permu_static", soft_permu)
        print(tf.reduce_sum(soft_permu, axis=-1))
        print(tf.reduce_sum(soft_permu, axis=-2))
        y = soft_permu[:,0,0]
    print(tape.gradient(y, x))

    with tf.GradientTape() as tape:
        tape.watch(x)
        soft_permu = sinkhorn_eager(x)
        print("soft_permu_eager", soft_permu)
        y = soft_permu[:,0,0]
    print(tape.gradient(y, x))

from voi.nn.base.sinkhorn import *
import tensorflow as tf
import tensorflow_probability as tfp
import time

if __name__ == "__main__":

    @tf.function
    def calc(x):
        print("tracing")
        with tf.GradientTape() as tape:
            tape.watch(x)
            soft_permu = skh.call(x)
            y = tf.linalg.trace(tf.matmul(x, soft_permu, transpose_a=True)) - tf.reduce_sum(soft_permu * tf.math.log(soft_permu))
        #tf.print("y calc1", y)
        dydx = tape.gradient(y, x)
        return dydx

    @tf.function
    def calc2(x):
        print("tracing")
        with tf.GradientTape() as tape:
            tape.watch(x)
            soft_permu = tf.stop_gradient(skh.call(x))
            y = tf.linalg.trace(tf.matmul(x, soft_permu, transpose_a=True)) - tf.reduce_sum(soft_permu * tf.math.log(soft_permu))
        #tf.print("y calc2", y)
        dydx = tape.gradient(y, x)
        return dydx

    tf.random.set_seed(1) 
    n = 10
    x = tf.random.normal([1, n, n]) * 10.0
    skh = Sinkhorn(200)
    print("original x", x)

    """
    with tf.GradientTape() as tape:
        tape.watch(x)
        soft_permu = skh.call(x)
        y = tf.linalg.trace(tf.matmul(x, soft_permu, transpose_a=True)) - tf.reduce_sum(soft_permu * tf.math.log(soft_permu))
    dydx1 = tape.gradient(y, x)
    #print(dydx1)

    with tf.GradientTape() as tape:
        tape.watch(x)
        soft_permu = tf.stop_gradient(skh.call(x))
        y = tf.linalg.trace(tf.matmul(x, soft_permu, transpose_a=True)) - tf.reduce_sum(soft_permu * tf.math.log(soft_permu))
    dydx2 = tape.gradient(y, x)
    #print(dydx2)
    print(tf.math.abs(dydx1-dydx2))
    """


    cur_time = time.time()
    for _ in range(50):
        x = tf.random.normal([1, n, n]) * 3
        dydx3 = calc(x)
    print("Time:", time.time() - cur_time)

    cur_time = time.time()
    for _ in range(50):
        x = tf.random.normal([1, n, n]) * 3
        dydx4 = calc2(x)
    print("Time:", time.time() - cur_time)
    #print(tf.math.abs(dydx3-dydx4))

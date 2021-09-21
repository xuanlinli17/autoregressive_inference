from voi.nn.base.sinkhorn import *
import tensorflow as tf
import tensorflow_probability as tfp

if __name__ == "__main__":
    sz = 20
    n = 1
    g = tfp.distributions.Gumbel(loc=tf.zeros([n,sz,sz]), scale=tf.ones([n,sz,sz]) * 50.0)
    sampled_noise = g.sample()
    print(sampled_noise)
    temperature = 1.0

    skh = Sinkhorn(200)

    soft_permu = skh.call(sampled_noise)
    hard_permu = matching(soft_permu)
    for i in range(n):
        print(hard_permu[i])
    print(tf.reduce_sum(soft_permu, axis=(-2)))
    print(tf.reduce_sum(soft_permu, axis=(-1)))
"""
    y = tf.reduce_sum(p * c[..., tf.newaxis, tf.newaxis], axis=1)

    print(x[0])
    print(y[0])

    print(tf.reduce_sum(y, axis=-1)[0])

    print(tf.reduce_sum(y, axis=-2)[0])
"""

import tensorflow as tf
import numpy as np

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4):
        self.mean = tf.Variable(0.0, trainable=False)
        self.var = tf.Variable(1.0, trainable=False)
        self.count = tf.Variable(epsilon, trainable=False)

    #@tf.function
    def update(self, x):
        batch_mean = tf.reduce_mean(x, axis=0)
        batch_var = tf.math.reduce_std(x, axis=0) ** 2
        # We don't want gradient to propagate back through self.mean and self.var
        batch_mean = tf.stop_gradient(batch_mean)
        batch_var = tf.stop_gradient(batch_var)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        if self.count >= 9999:
            self.count.assign_sub(batch_count)
        tot_count = self.count + batch_count
        self.mean.assign_add(delta * batch_count / tot_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + tf.square(delta) * self.count * batch_count / tot_count
        self.var.assign(M2 / tot_count)
        self.count.assign(tot_count)
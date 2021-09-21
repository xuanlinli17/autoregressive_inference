import tensorflow as tf
import numpy as np

def bethe_loop_fn(logV1,
                  logV2,
                  step,
                  iterations):
    # http://proceedings.mlr.press/v118/mena20a/mena20a.pdf
    
    eps = tf.constant(1e-20)
    logexpV2 = tf.math.log(-tf.math.expm1(logV2)+eps)
    HelpMat = logV2 + logexpV2
    HelpMat = HelpMat - tf.math.log(-tf.math.expm1(logV2)+eps)
    logV1 = HelpMat - tf.math.reduce_logsumexp(HelpMat,1,keepdims=True)
    HelpMat = logV1 + logexpV2
    HelpMat = HelpMat - tf.math.log(-tf.math.expm1(logV1)+eps)
    logV2 = HelpMat - tf.math.reduce_logsumexp(HelpMat,2,keepdims=True)
    return logV1, logV2, step + 1, iterations


def bethe_cond_fn(logV1,
                  logV2,
                  step,
                  iterations):

    return tf.less(step, iterations)

# Tensorflow will hang if @tf.function here for some reason
# @tf.function(input_signature=[
#     tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
#     tf.TensorSpec(shape=None, dtype=tf.int32)])
def bethe(x,
          iterations):
    """Calculate the result of applying the Bethe Operator
    to a permutation matrix in log space

    Arguments:

    x: tf.Tensor
        a permutation matrix in log space that will be
        processed with the bethe operator
    iterations: tf.Tensor
        the total number of iterations of the Bethe operator
        to apply to the data matrix

    Returns:

    x: tf.Tensor
        a permutation matrix in log space that has been
        processed with the bethe operator"""
    
    bs, N = tf.shape(x)[0], tf.shape(x)[1]
    logV1 = tf.math.log(tf.cast(1/N, tf.float32) * tf.ones([bs,N,N]))
    logV2 = x - tf.math.reduce_logsumexp(x, axis=2, keepdims=True)

    args = [logV1, logV2, tf.constant(0, dtype=tf.int32), iterations]
    return tf.while_loop(
        bethe_cond_fn, bethe_loop_fn, args)[0]


class Bethe(tf.keras.layers.Layer):

    def __init__(self,
                 iterations=20):
        """Calculate the result of applying the Bethe Operator
        to a permutation matrix in log space

        Arguments:

        iterations: tf.Tensor
            the total number of iterations of the Bethe operator
            to apply to the data matrix"""
        super(Bethe, self).__init__()

        self.iterations = iterations

    def call(self, inputs, **kwargs):
        # apply the bethe operator
        inputs = tf.stop_gradient(inputs)
        return tf.exp(bethe(inputs, tf.constant(self.iterations)))
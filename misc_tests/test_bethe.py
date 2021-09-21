def bethe_loop_fn(logV1,
                  logV2,
                  step,
                  iterations):
    """Calculate the result of applying the Bethe Operator
    to a permutation matrix in log space

    Arguments:

    x: tf.Tensor
        a permutation matrix in log space that will be
        processed with the bethe operator
    step: tf.Tensor
        the current number of iterations of the Bethe operator
        that have been applied
    iterations: tf.Tensor
        the total number of iterations of the Bethe operator
        to apply to the data matrix

    Returns:

    x: tf.Tensor
        a permutation matrix in log space that has been
        processed with the bethe operator
    step: tf.Tensor
        the current number of iterations of the Bethe operator
        that have been applied
    iterations: tf.Tensor
        the total number of iterations of the Bethe operator
        to apply to the data matrix"""
    
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
    """Calculate the result of applying the Bethe Operator
    to a permutation matrix in log space

    Arguments:

    x: tf.Tensor
        a permutation matrix in log space that will be
        processed with the bethe operator
    step: tf.Tensor
        the current number of iterations of the Bethe operator
        that have been applied
    iterations: tf.Tensor
        the total number of iterations of the Bethe operator
        to apply to the data matrix

    Returns:

    condition: tf.Tensor
        a boolean that determines if the loop that applies
        the Bethe Operator should exit"""

    return tf.less(step, iterations)


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

import tensorflow as tf

n = 20
a = tf.random.normal([2, 20, 20]) * 5.0
print(a)
a = tf.exp(bethe(a, tf.constant(50)))
print(a)
tf.print(tf.reduce_sum(a, axis=-2), tf.reduce_sum(a, axis=-1), summarize=-1)

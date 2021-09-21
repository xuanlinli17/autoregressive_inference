import tensorflow as tf
import numpy as np
from scipy.optimize import linear_sum_assignment
import time

# This uses tf.py_function to call scipy.optimize.linear_sum_assignment,
# which is slow in multi-gpu setting (as tf only allows one py_function to run
# in the address space of the program
def hungarian(x):
    if x.ndim == 2:
        x = np.reshape(x, [1, x.shape[0], x.shape[1]])
    sol = np.zeros((x.shape[0], x.shape[1]), dtype=np.int32)
    for i in range(x.shape[0]):
        sol[i, :] = linear_sum_assignment(x[i, :])[1].astype(np.int32)
    return sol

def matching(matrix_batch):
    """Solves a matching problem for a batch of matrices.
    Modified from 
    https://github.com/google/gumbel_sinkhorn/blob/master/sinkhorn_ops.py
    
    This is a wrapper for the scipy.optimize.linear_sum_assignment function. It
    solves the optimization problem min_P sum_i,j M_i,j P_i,j with P a
    permutation matrix.
    Args:
    matrix_batch: A 3D tensor (a batch of matrices) with
      shape = [batch_size, N, N]. If 2D, the input is reshaped to 3D with
      batch_size = 1.
    Returns:
    listperms, a 2D integer tensor of permutations with shape [batch_size, N]
      so that listperms[n, :] can be written as a permutation matrix P of size N*N that solves the
      problem  min_P sum_i,j M_i,j P_i,j with M = matrix_batch[n, :, :].
    """

    listperms = tf.py_function(func=hungarian, inp=[matrix_batch], Tout=tf.int32) # returns 2D
    listperms.set_shape(tf.TensorShape([None, None]))
    return listperms

def sinkhorn_loop_fn(x,
                     step,
                     iterations):
    """Calculate the result of applying the Sinkhorn Operator
    to a permutation matrix in log space

    Arguments:

    x: tf.Tensor
        a permutation matrix in log space that will be
        processed with the sinkhorn operator
    step: tf.Tensor
        the current number of iterations of the Sinkhorn operator
        that have been applied
    iterations: tf.Tensor
        the total number of iterations of the Sinkhorn operator
        to apply to the data matrix

    Returns:

    x: tf.Tensor
        a permutation matrix in log space that has been
        processed with the sinkhorn operator
    step: tf.Tensor
        the current number of iterations of the Sinkhorn operator
        that have been applied
    iterations: tf.Tensor
        the total number of iterations of the Sinkhorn operator
        to apply to the data matrix"""

    x = tf.math.log_softmax(x, axis=-2)
    x = tf.math.log_softmax(x, axis=-1)
    return x, step + 1, iterations


def sinkhorn_cond_fn(x,
                     step,
                     iterations):
    """Calculate the result of applying the Sinkhorn Operator
    to a permutation matrix in log space

    Arguments:

    x: tf.Tensor
        a permutation matrix in log space that will be
        processed with the sinkhorn operator
    step: tf.Tensor
        the current number of iterations of the Sinkhorn operator
        that have been applied
    iterations: tf.Tensor
        the total number of iterations of the Sinkhorn operator
        to apply to the data matrix

    Returns:

    condition: tf.Tensor
        a boolean that determines if the loop that applies
        the Sinkhorn Operator should exit"""

    return tf.less(step, iterations)


# @tf.function(input_signature=[
#     tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
#     tf.TensorSpec(shape=None, dtype=tf.int32)])
def sinkhorn(x,
             iterations):
    """Calculate the result of applying the Sinkhorn Operator
    to a permutation matrix in log space

    Arguments:

    x: tf.Tensor
        a permutation matrix in log space that will be
        processed with the sinkhorn operator
    iterations: tf.Tensor
        the total number of iterations of the Sinkhorn operator
        to apply to the data matrix

    Returns:

    x: tf.Tensor
        a permutation matrix in log space that has been
        processed with the sinkhorn operator"""

    args = [x, tf.constant(0, dtype=tf.int32), iterations]
    return tf.while_loop(
        sinkhorn_cond_fn, sinkhorn_loop_fn, args)[0]


class Sinkhorn(tf.keras.layers.Layer):

    def __init__(self,
                 iterations=20):
        """Calculate the result of applying the Sinkhorn Operator
        to a permutation matrix in log space

        Arguments:

        iterations: tf.Tensor
            the total number of iterations of the Sinkhorn operator
            to apply to the data matrix"""
        super(Sinkhorn, self).__init__()

        self.iterations = iterations

    def call(self, inputs, **kwargs):
        # apply the sinkhorn operator
        return tf.exp(sinkhorn(inputs, tf.constant(self.iterations)))

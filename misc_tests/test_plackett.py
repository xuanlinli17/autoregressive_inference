import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from scipy.optimize import linear_sum_assignment

def hungarian(x):
    if x.ndim == 2:
        x = np.reshape(x, [1, x.shape[0], x.shape[1]])
    sol = np.zeros((x.shape[0], x.shape[1]), dtype=np.int32)
    for i in range(x.shape[0]):
        sol[i, :] = linear_sum_assignment(-x[i, :])[1].astype(np.int32)
    return sol

def deterministic_NeuralSort(logs, tau, mask):
    """s: input elements to be sorted. Shape: batch_size x n x 1
    tau: temperature for relaxation. Scalar."""
    s = tf.math.exp(logs) * mask
    tf.print("s[0]", tf.squeeze(s[0]), summarize=-1)
    n = tf.shape(s)[1]
#     even_n = False
#     if n % 2 == 0:
#         even_n = True
#         n += 1
#         s = tf.concat([s, tf.zeros([tf.shape(s)[0], 1, 1])], axis=1)
#         mask = tf.concat([mask, tf.zeros([tf.shape(s)[0], 1, 1])], axis=1)
    one = tf.ones((n, 1), dtype = tf.float32)
    A_s = tf.math.abs(s - tf.transpose(s, [0, 2, 1]))
    B = tf.matmul(A_s, tf.matmul(one, tf.transpose(one)))
    scaling = tf.cast(n + 1 - 2*(tf.range(n) + 1), dtype = tf.float32)
    C = tf.matmul(s, tf.expand_dims(scaling, 0))
    P_max = tf.transpose(C-B, perm=[0, 2, 1])
    P_hat = tf.nn.softmax(P_max / tau, -1)
    P_hat = P_hat * tf.transpose(mask, (0,2,1))
#     if even_n:
#         P_hat = P_hat[:, :-2, :] # 0 is always masked out, which is always placed in the last
#     else:
#         P_hat = P_hat[:, :-1, :]
    #P_hat = P_hat[:, :-1, :]
    #tf.print("phat", P_hat[0], summarize=-1)
    return P_hat  

@tf.function
def matching2d(matrix_batch):
    """Solves a matching problem for a batch of matrices.
    Modified from 
    https://github.com/google/gumbel_sinkhorn/blob/master/sinkhorn_ops.py
    
    This is a wrapper for the scipy.optimize.linear_sum_assignment function. It
    solves the optimization problem max_P sum_i,j M_i,j P_i,j with P a
    permutation matrix. Notice the negative sign; the reason, the original
    function solves a minimization problem
    Args:
    matrix_batch: A 3D tensor (a batch of matrices) with
      shape = [batch_size, N, N]. If 2D, the input is reshaped to 3D with
      batch_size = 1.
    Returns:
    listperms, a 2D integer tensor of permutations with shape [batch_size, N]
      so that listperms[n, :] is the permutation matrix P of size N*N that solves the
      problem  max_P sum_i,j M_i,j P_i,j with M = matrix_batch[n, :, :].
    """

    listperms = tf.py_function(func=hungarian_shuffle, inp=[matrix_batch, [16]], Tout=tf.int32) # 2D
    listperms.set_shape(tf.TensorShape([None, None]))
    return listperms

def hungarian_shuffle(x, mask):
    if x.ndim == 2:
        x = np.reshape(x, [1, x.shape[0], x.shape[1]])
    sol = np.zeros((x.shape[0], x.shape[1]), dtype=np.int32)
    for i in range(x.shape[0]):
        sol[i, :] = linear_sum_assignment(-x[i, :])[1].astype(np.int32)
        tmp = sol[i, :]
        criterion = np.logical_and(tmp>0, tmp<=mask[i])
        sol[i, :] = np.concatenate([tmp[criterion],
                                    tmp[np.logical_not(criterion)]])
    return sol

logs = tf.ones([1,20,1])    * 1e-2
#noise = tf.random.uniform(shape=tf.shape(logs), maxval=1e-2)
mask = tf.concat([tf.zeros([1,1,1]), tf.ones([1,16,1]), tf.zeros([1,3,1])], axis=1)

"""
logs = tf.constant([0.01, 1.0, 0.0099])
logs = tf.reshape(logs, [1,3,1])
noise = tf.random.uniform(shape=tf.shape(logs), maxval=1e-2)
mask = tf.ones_like(logs)
"""
result = deterministic_NeuralSort(logs, 1.0, mask)
print(result)
print((result - (1-tf.transpose(mask, [0,2,1])) * 100000.0))
result = matching2d(-(result - (1-tf.transpose(mask, [0,2,1])) * 100000.0))
print(result)
print(tf.cast(tf.math.argmax(result, axis=-1), tf.int32))

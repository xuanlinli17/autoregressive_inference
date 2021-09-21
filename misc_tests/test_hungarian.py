from voi.nn.base.sinkhorn import matching
from scipy.optimize import linear_sum_assignment
import tensorflow as tf
import numpy as np

def spy(x):
    if x.ndim == 2:
        x = np.reshape(x, [1, x.shape[0], x.shape[1]])
    sol = np.zeros((x.shape[0], x.shape[1]), dtype=np.int32)
    for i in range(x.shape[0]):
        sol[i, :] = linear_sum_assignment(-x[i, :])[1].astype(np.int32)
    return sol

#@tf.function
def wrap(x):
    result = matching(x)
    return tf.reshape(tf.where(result == 1), [128, 80, 3])[:, :, -1]

x = tf.random.normal([128, 80, 80])
print("scipy:", spy(x))
print("matching:", wrap(x))

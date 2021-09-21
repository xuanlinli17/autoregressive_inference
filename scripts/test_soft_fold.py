from voi.nn.base.stick_breaking import stick_breaking, inv_stick_breaking
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def fold_loop_fn(p, a, step):
    """Update function for a differentiable fold operation that
    maps doubly stochastic matrices into lower triangular
    row stochastic matrices

    Arguments:

    p: tf.Tensor
        the lower triangular row-stochastic matrix that is to be
        filled in by the loop function
    a: tf.Tensor
        the doubly-stochastic matrix that is used to calculate values in
        the lower triangular row-stochastic matrix
    step: tf.Tensor
        the integer step of the loop function, runs for N^2 updates
        where N is the height of the doubly stochastic matrix

    Returns:

    p: tf.Tensor
        the lower triangular row-stochastic matrix that is to be
        filled in by the loop function
    a: tf.Tensor
        the doubly-stochastic matrix that is used to calculate values in
        the lower triangular row-stochastic matrix
    step: tf.Tensor
        the integer step of the loop function, runs for N^2 updates
        where N is the height of the doubly stochastic matrix"""

    i = tf.math.floordiv(step, tf.shape(a)[2])
    j = tf.math.floormod(step, tf.shape(a)[2])
    N = tf.shape(a)[2]

    # check the probability that w was decoded at step i
    p_token_w_step_i = a[..., i, :, tf.newaxis]

    tf.print(i, j, p_token_w_step_i, summarize=-1)

    # re-weight the probabilities to remove token w from the distribution
    # this corresponds to conditioning on w already being sampled
    p_token_v_slot_j = tf.repeat(a[..., j, tf.newaxis, :], N, axis=-2)
    p_token_v_slot_j = tf.math.divide_no_nan(
        p_token_v_slot_j, 1.0 - a[..., j, :, tf.newaxis])

    # assign zero probabilities to where v == w
    # this corresponds to conditioning on w already being sampled
    p_token_v_slot_j = p_token_v_slot_j * (
        1.0 - tf.eye(N, batch_shape=tf.shape(a)[:-2]))

    tf.print(i, j, p_token_v_slot_j, summarize=-1)

    # re-weight to remove tokens w and v from the distribution
    # this corresponds to conditioning on w and v already being sampled
    p_step_i_slot_j = tf.reduce_sum(a[..., i + 1:, :], axis=-2)

    tf.print(p_step_i_slot_j, a[..., j, :], a[..., j, :], summarize=-1)
    p_step_i_slot_j = tf.math.divide_no_nan(
        p_step_i_slot_j, 1.0 - a[..., j, :] - a[..., i, :])

    z = tf.range(N)
    x, y = tf.meshgrid(z, z, indexing='ij')
    mask = tf.cast(tf.logical_and(tf.greater(
        z[tf.newaxis, tf.newaxis, :], y[:, :, tf.newaxis]), tf.less(
        z[tf.newaxis, tf.newaxis, :], x[:, :, tf.newaxis])), tf.float32)

    # crop probabilities to only those between w and v

    tf.print(mask, summarize=-1)

    tf.print(i, j, p_step_i_slot_j, summarize=-1)
    p_step_i_slot_j = p_step_i_slot_j[..., tf.newaxis, tf.newaxis, :]
    p_step_i_slot_j = tf.repeat(p_step_i_slot_j, N, axis=-3)
    p_step_i_slot_j = tf.repeat(p_step_i_slot_j, N, axis=-2)
    p_step_i_slot_j = p_step_i_slot_j * mask + 1.0 - mask
    p_step_i_slot_j = tf.reduce_prod(p_step_i_slot_j, axis=-1)

    x, y = tf.meshgrid(tf.range(N), tf.range(N), indexing='ij')
    p_joint = tf.cast(tf.less([x], [y]), tf.float32) * (
        p_token_w_step_i * p_token_v_slot_j * p_step_i_slot_j)
    p_joint = tf.where(tf.less(j, i), p_joint, tf.zeros_like(p_joint))

    mask = tf.logical_and(tf.equal(i, [x]), tf.equal(j, [y]))
    return tf.where(mask, p_joint, p), a, step + 1


def fold_cond_fn(p, a, step):
    """Update function for a differentiable fold operation that
    maps doubly stochastic matrices into lower triangular
    row stochastic matrices

    Arguments:

    p: tf.Tensor
        the lower triangular row-stochastic matrix that is to be
        filled in by the loop function
    a: tf.Tensor
        the doubly-stochastic matrix that is used to calculate values in
        the lower triangular row-stochastic matrix
    step: tf.Tensor
        the integer step of the loop function, runs for N^2 updates
        where N is the height of the doubly stochastic matrix

    Returns:

    p: tf.Tensor
        the lower triangular row-stochastic matrix that is to be
        filled in by the loop function
    a: tf.Tensor
        the doubly-stochastic matrix that is used to calculate values in
        the lower triangular row-stochastic matrix
    step: tf.Tensor
        the integer step of the loop function, runs for N^2 updates
        where N is the height of the doubly stochastic matrix"""

    return tf.less(step, tf.square(tf.shape(a)[2]))


@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, None, None], dtype=tf.float32)])
def fold(a):
    """Calculate the result of applying the Stick Breaking Operator
    to a permutation matrix in logistic space

    Arguments:

    a: tf.Tensor
        the doubly-stochastic matrix that is used to calculate values in
        the lower triangular row-stochastic matrix

    Returns:

    p: tf.Tensor
        the lower triangular row-stochastic matrix that is to be
        filled in by the loop function"""

    args = [tf.zeros_like(a), a, tf.constant(0, dtype=tf.int32)]
    return tf.while_loop(fold_cond_fn, fold_loop_fn, args)[0]


if __name__ == "__main__":

    logits = tf.random.normal([1, 12, 12]) / 100.0
    mask = tf.ones_like(logits)

    x = mask / tf.reduce_sum(mask, axis=1, keepdims=True)
    z = inv_stick_breaking(x, mask)

    a = stick_breaking(tf.math.sigmoid(logits -
                                       tf.math.log(1. / z - 1.)), mask)

    x = fold(a)

    print(x)

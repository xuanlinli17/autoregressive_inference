import tensorflow as tf


def stick_breaking_loop_fn(x,
                           x_mask,
                           b,
                           step):
    """Calculate the result of applying the Stick Breaking Operator
    to a permutation matrix in log space

    Arguments:

    x: tf.Tensor
        a permutation matrix that will be generated using the stick
        breaking procedure
    x_mask: tf.Tensor
        a mask that specifies which elements of the permutation
        matrix are allowed to be non-zero
    b: tf.Tensor
        a permutation matrix in logistic space that will be
        processed with the Stick Breaking operator
    step: tf.Tensor
        the current number of iterations of the Stick Breaking operator
        that have been applied=

    Returns:

    x: tf.Tensor
        a permutation matrix that will be generated using the stick
        breaking procedure
    x_mask: tf.Tensor
        a mask that specifies which elements of the permutation
        matrix are allowed to be non-zero
    b: tf.Tensor
        a permutation matrix in logistic space that will be
        processed with the Stick Breaking operator
    step: tf.Tensor
        the current number of iterations of the Stick Breaking operator
        that have been applied"""

    m = tf.math.floordiv(step, tf.shape(b)[2])
    n = tf.math.floormod(step, tf.shape(b)[2])
    N = tf.shape(b)[2]

    max_future_vals = tf.maximum(0., x_mask[
        :, m, (n + 1):] - tf.reduce_sum(x[:, :m, (n + 1):], axis=1))
    max_future_mass = tf.reduce_sum(max_future_vals, axis=1)

    lower_bound = x_mask[:, m, n] * tf.maximum(
        0.,
        1. - tf.reduce_sum(x[:, m, :n], axis=1) - max_future_mass)
    upper_bound = x_mask[:, m, n] * tf.minimum(
        1. - tf.reduce_sum(x[:, m, :n], axis=1),
        1. - tf.reduce_sum(x[:, :m, n], axis=1))

    lower_bound = tf.minimum(lower_bound, upper_bound)
    p = lower_bound + b[:, m, n] * (upper_bound - lower_bound)
    p = tf.clip_by_value(
        p, 0., 1.)[:, tf.newaxis, tf.newaxis]

    i, j = tf.meshgrid(tf.range(N), tf.range(N), indexing='ij')
    mask = tf.logical_and(
        tf.equal(i, [[m]]), tf.equal(j, [[n]]))[tf.newaxis]

    return tf.where(mask, p, x), x_mask, b, step + 1


def stick_breaking_cond_fn(x,
                           x_mask,
                           b,
                           step):
    """Calculate the result of applying the Stick Breaking Operator
    to a permutation matrix in log space

    Arguments:

    x: tf.Tensor
        a permutation matrix that will be generated using the stick
        breaking procedure
    x_mask: tf.Tensor
        a mask that specifies which elements of the permutation
        matrix are allowed to be non-zero
    b: tf.Tensor
        a permutation matrix in logistic space that will be
        processed with the Stick Breaking operator
    step: tf.Tensor
        the current number of iterations of the Stick Breaking operator
        that have been applied

    Returns:

    x: tf.Tensor
        a permutation matrix that will be generated using the stick
        breaking procedure
    x_mask: tf.Tensor
        a mask that specifies which elements of the permutation
        matrix are allowed to be non-zero
    b: tf.Tensor
        a permutation matrix in logistic space that will be
        processed with the Stick Breaking operator
    step: tf.Tensor
        the current number of iterations of the Stick Breaking operator
        that have been applied"""

    return tf.less(step, tf.square(tf.shape(b)[2]))


# @tf.function(input_signature=[
#     tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
#     tf.TensorSpec(shape=[None, None, None], dtype=tf.float32)])
def stick_breaking(x,
                   x_mask):
    """Calculate the result of applying the Stick Breaking Operator
    to a permutation matrix in logistic space

    Arguments:

    x: tf.Tensor
        a permutation matrix in log space that will be
        processed with the Stick Breaking operator
    x_mask: tf.Tensor
        a mask that specifies which elements of the permutation
        matrix are allowed to be non-zero

    Returns:

    x: tf.Tensor
        a permutation matrix in log space that has been
        processed with the Stick Breaking operator
    log_det_jac: tf.Tensor
        the log determinant of the jacobian of the stick breaking
        transformation for evaluating the entropy"""

    args = [tf.zeros_like(x), x_mask, x, tf.constant(0, dtype=tf.int32)]
    return tf.while_loop(
        stick_breaking_cond_fn, stick_breaking_loop_fn, args)[0]


def inv_stick_breaking_loop_fn(x,
                               x_mask,
                               b,
                               step):
    """Calculate the result of applying the Stick Breaking Operator
    to a permutation matrix in log space

    Arguments:

    x: tf.Tensor
        a permutation matrix that will be generated using the stick
        breaking procedure
    x_mask: tf.Tensor
        a mask that specifies which elements of the permutation
        matrix are allowed to be non-zero
    b: tf.Tensor
        a permutation matrix in logistic space that will be
        processed with the Stick Breaking operator
    step: tf.Tensor
        the current number of iterations of the Stick Breaking operator
        that have been applied

    Returns:

    x: tf.Tensor
        a permutation matrix that will be generated using the stick
        breaking procedure
    x_mask: tf.Tensor
        a mask that specifies which elements of the permutation
        matrix are allowed to be non-zero
    b: tf.Tensor
        a permutation matrix in logistic space that will be
        processed with the Stick Breaking operator
    step: tf.Tensor
        the current number of iterations of the Stick Breaking operator
        that have been applied"""

    m = tf.math.floordiv(step, tf.shape(b)[2])
    n = tf.math.floormod(step, tf.shape(b)[2])
    N = tf.shape(b)[2]

    max_future_vals = tf.maximum(0., x_mask[
        :, m, (n + 1):] - tf.reduce_sum(x[:, :m, (n + 1):], axis=1))
    max_future_mass = tf.reduce_sum(max_future_vals, axis=1)

    lower_bound = x_mask[:, m, n] * tf.maximum(
        0.,
        1. - tf.reduce_sum(x[:, m, :n], axis=1) - max_future_mass)
    upper_bound = x_mask[:, m, n] * tf.minimum(
        1. - tf.reduce_sum(x[:, m, :n], axis=1),
        1. - tf.reduce_sum(x[:, :m, n], axis=1))

    lower_bound = tf.minimum(lower_bound, upper_bound)
    p = tf.math.divide_no_nan(
        x[:, m, n] - lower_bound, upper_bound - lower_bound)
    p = tf.clip_by_value(
        p, 1e-7, 1 - 1e-7)[:, tf.newaxis, tf.newaxis]

    i, j = tf.meshgrid(tf.range(N), tf.range(N), indexing='ij')
    mask = tf.logical_and(
        tf.equal(i, [[m]]), tf.equal(j, [[n]]))[tf.newaxis]

    return x, x_mask, tf.where(mask, p, b), step + 1


def inv_stick_breaking_cond_fn(x,
                               x_mask,
                               b,
                               step):
    """Calculate the result of applying the Stick Breaking Operator
    to a permutation matrix in log space

    Arguments:

    x: tf.Tensor
        a permutation matrix that will be generated using the stick
        breaking procedure
    x_mask: tf.Tensor
        a mask that specifies which elements of the permutation
        matrix are allowed to be non-zero
    b: tf.Tensor
        a permutation matrix in logistic space that will be
        processed with the Stick Breaking operator
    step: tf.Tensor
        the current number of iterations of the Stick Breaking operator
        that have been applied

    Returns:

    x: tf.Tensor
        a permutation matrix that will be generated using the stick
        breaking procedure
    x_mask: tf.Tensor
        a mask that specifies which elements of the permutation
        matrix are allowed to be non-zero
    b: tf.Tensor
        a permutation matrix in logistic space that will be
        processed with the Stick Breaking operator
    step: tf.Tensor
        the current number of iterations of the Stick Breaking operator
        that have been applied"""

    return tf.less(step, tf.square(tf.shape(x)[2]))


# @tf.function(input_signature=[
#     tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
#     tf.TensorSpec(shape=[None, None, None], dtype=tf.float32)])
def inv_stick_breaking(x,
                       x_mask):
    """Calculate the result of applying the Stick Breaking Operator
    to a permutation matrix in logistic space

    Arguments:

    x: tf.Tensor
        a permutation matrix in log space that will be
        processed with the Stick Breaking operator
    x_mask: tf.Tensor
        a mask that specifies which elements of the permutation
        matrix are allowed to be non-zero

    Returns:

    x: tf.Tensor
        a permutation matrix in log space that has been
        processed with the Stick Breaking operator
    log_det_jac: tf.Tensor
        the log determinant of the jacobian of the stick breaking
        transformation for evaluating the entropy"""

    args = [x, x_mask, tf.zeros_like(x), tf.constant(0, dtype=tf.int32)]
    return tf.while_loop(
        inv_stick_breaking_cond_fn, inv_stick_breaking_loop_fn, args)[2]


class StickBreaking(tf.keras.layers.Layer):

    def __init__(self):
        """Calculate the result of applying the Stick Breaking Operator
        to a permutation matrix in log space"""
        super(StickBreaking, self).__init__()

    def call(self, inputs, **kwargs):
        """Runs a forward pass on a pointer network that generates
        permutation matrices in logistic space

        Arguments:

        inputs: list of Tensors

        Returns:

        outputs: tf.Tensor
            a permutation matrix in logistic space that has the same shape
            as the transformer attention weights"""

        # calculate the z that leads to a uniform doubly stochastic matrix
        # let this z be the center of the stick breaking logits
        x = inputs[1] / tf.reduce_sum(inputs[1], axis=1, keepdims=True)
        z = inv_stick_breaking(x, inputs[1])

        # in addition, account for the greater sensitivity to the operation
        # to large values to the upper left of the logits matrix
        return stick_breaking(tf.math.sigmoid(
            inputs[0] - tf.math.log(1. / z - 1.)), inputs[1])
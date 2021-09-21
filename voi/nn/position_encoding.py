import tensorflow as tf
import math


def position_encoding(length,
                      hidden_size,
                      min_timescale=1.0,
                      max_timescale=1.0e4):
    """Return positional encoding.

    See: https://github.com/tensorflow/models/blob/master/
         official/nlp/transformer/model_utils.py

    Calculates the position encoding as a mix of sine and cosine functions with
    geometrically increasing wavelengths.
    Defined and formalized in Attention is All You Need, section 3.5.

    Args:

    length: int or tf.Tensor
        Sequence length.
    hidden_size: int
        Size of the
    min_timescale: float
        Minimum scale that will be applied at each position
    max_timescale: float
        Maximum scale that will be applied at each position

    Returns:

    Tensor with shape [length, hidden_size]
    """

    position = tf.cast(tf.range(length), tf.float32)
    num_timescales = hidden_size // 2

    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (tf.cast(num_timescales, tf.float32) - 1))

    inv_timescales = min_timescale * tf.exp(
        tf.cast(tf.range(num_timescales),
                tf.float32) * -log_timescale_increment)

    scaled_time = tf.expand_dims(
        position, 1) * tf.expand_dims(inv_timescales, 0)

    return tf.concat([tf.sin(scaled_time),
                      tf.cos(scaled_time)], axis=1)

def position_encoding_relative(length,
                               hidden_size,
                               min_timescale=1.0,
                               max_timescale=1.0e4):
    """Return positional encoding similar to position_encoding,
    except that the positions are now relative positions, in the range
    of [-length, length]. This is used for Transformer-XL based position
    encoding.
    
    Returns:

    Tensor with shape [2*length + 1, hidden_size]
    """

    position = tf.cast(tf.range(-length, length + 1), tf.float32)
    num_timescales = hidden_size // 2

    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (tf.cast(num_timescales, tf.float32) - 1))

    inv_timescales = min_timescale * tf.exp(
        tf.cast(tf.range(num_timescales),
                tf.float32) * -log_timescale_increment)

    scaled_time = tf.expand_dims(
        position, 1) * tf.expand_dims(inv_timescales, 0)

    return tf.concat([tf.sin(scaled_time),
                      tf.cos(scaled_time)], axis=1)

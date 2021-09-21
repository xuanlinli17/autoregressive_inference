import tensorflow as tf

def custom_gather(a, b):
    """ Permutes the vectors in the last dimension of a according to
        the corresponding vectors in the last dimension of b
        e.g. a=[[1,4,7,6],[4,3,6,8]], b=[[0,3,1,2],[2,1,3,0]]
        Returns [[1,6,4,7],[6,3,8,4]]
    """
    assert len(tf.shape(a)) >= 2 # first dimension is batch size
    assert len(tf.shape(a)) == len(tf.shape(b))
    original_shape = tf.shape(a)
    lastdim = tf.shape(a)[-1]
    a = tf.reshape(a, (-1, lastdim))
    b = tf.reshape(b, (-1, lastdim))
    idx = tf.range(tf.shape(a)[0])[:, tf.newaxis]
    idx = tf.tile(idx, [1, tf.shape(a)[1]])
    idx = tf.concat([idx[..., tf.newaxis], b[..., tf.newaxis]], axis=-1)
    result = tf.gather_nd(a, idx)
    result = tf.reshape(result, original_shape)
    return result   
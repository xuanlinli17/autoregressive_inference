import tensorflow as tf


class SequenceToMatSinkhorn(tf.keras.layers.Layer):

    def __init__(self,
                 queries_dropout=0.,
                 keys_dropout=0.):
        """Creates the backbone for the logits of a permutation layer
        converts a sequence to a matrix of permutation logits
        Arguments:
        queries_dropout: float
            the ratio of units to drop during training to the
            number of units in each attention layer
        keys_dropout: float
            the ratio of units to drop during training to the
            number of units in each attention layer"""
        super(SequenceToMatSinkhorn, self).__init__()

        self.q_dropout = tf.keras.layers.Dropout(queries_dropout)
        self.k_dropout = tf.keras.layers.Dropout(keys_dropout)

        self.queries_dropout_rate = queries_dropout
        self.keys_dropout_rate = keys_dropout

    def static_call(self, queries, keys,
                    queries_mask, values_mask, **kwargs):
        """Runs a forward pass on a multi head attention layer
        inputs is an instance of AttentionInput
        Arguments:
        inputs: queries, keys, and values along with masks
        Returns:
        outputs: tf.Tensor
            Shape [batch_dim, seq_dim, seq_dim], 
            the result of QK^T/sqrt(d) with appropriate masks applied"""

        # apply dropout to the queries keys and values tensor
        # requires all to be like [batch, heads, ]
        queries = self.q_dropout(queries, **kwargs)
        keys = self.k_dropout(keys, **kwargs)

        # compute the multi head soft attention weights using
        # scaled dot product attention
        size = tf.math.sqrt(
            tf.cast(tf.shape(queries)[-1], tf.float32))
        scores = tf.matmul(
            queries, keys, transpose_b=True) / size

        # if an attention bias is provided that add the attention bias
        # to the pre softmax scores matrix
#         if hasattr(inputs, 'bias') and inputs.bias is not None:
#             scores = scores + inputs.bias

        # apply a mask to the scores matrix so that only real
        # non terminal elements are permuted out of place
        mask = tf.expand_dims(values_mask, -2)
        mask = tf.logical_and(mask, tf.expand_dims(queries_mask, -1))

        # pad tokens should not be permuted and logits on the diagonal
        # for pad tokens should not be masked out; this is necessary because
        # a valid permutation matrix has rows and columns that sum to one,
        # even for rows that correspond to pad tokens
        shape = tf.shape(mask)
        mask = tf.logical_or(mask, tf.eye(
            shape[-2],
            num_columns=shape[-1], batch_shape=shape[:-2], dtype=tf.bool))

        # apply a boolean mask to the keys and values
        return tf.where(
            mask, scores, tf.fill(tf.shape(scores), -999999.))

    def call(self, inputs, **kwargs):
        """Calculates QK^T/sqrt(d) to be used as the logits for the 
        permutation layer.

        Arguments:

        inputs: AttentionInput
            a dataclass instance that
            contains queries, keys, and values along with masks

        Returns:

        outputs: tf.Tensor
            Shape [batch_dim, seq_dim, seq_dim], 
            the result of QK^T/sqrt(d) with appropriate masks applied"""

        return self.static_call(inputs.queries, inputs.keys,
                                inputs.queries_mask, inputs.values_mask,
                                **kwargs)
import tensorflow as tf
from voi.nn.base.attention import causal_mask

class AttentionWithBias(tf.keras.layers.Layer):

    def __init__(self,
                 queries_dropout=0.,
                 keys_dropout=0.,
                 values_dropout=0.,
                 causal=True):
        """Creates the backbone for multi headed attention
        and supports dropout on the attention mask

        Arguments:

        queries_dropout: float
            the ratio of units to drop during training to the
            number of units in each attention layer
        keys_dropout: float
            the ratio of units to drop during training to the
            number of units in each attention layer
        values_dropout: float
            the ratio of units to drop during training to the
            number of units in each attention layer
        causal: bool
            specifies is the transformer should decoding using
            a causal mask to preserve the auto regressive property"""
        super(AttentionWithBias, self).__init__()

        self.q_dropout = tf.keras.layers.Dropout(queries_dropout)
        self.k_dropout = tf.keras.layers.SpatialDropout2D(keys_dropout)
        self.v_dropout = tf.keras.layers.SpatialDropout2D(values_dropout)

        self.queries_dropout_rate = queries_dropout
        self.keys_dropout_rate = keys_dropout
        self.values_dropout_rate = values_dropout
        self.causal = causal

    def call(self, inputs, return_scores=False, **kwargs):
        """Runs a forward pass on a multi head attention layer
        inputs is an instance of AttentionInput

        Arguments:

        inputs: list of tf.Tensor
            contains queries, keys, and values along with masks

        Returns:

        outputs: tf.Tensor
            the result of applying a multi head attention mechanism
            will be shaped [batch_dim, seq_dim, channels]"""

        # unpack all the required model inputs
        [queries, keys, values, queries_mask, values_mask, bias] = inputs

        # apply dropout to the queries keys and values tensor
        # requires all to be like [batch, heads, ]
        queries = self.q_dropout(queries, **kwargs)
        keys = self.k_dropout(keys, **kwargs)
        values = self.v_dropout(values, **kwargs)

        # compute the multi head soft attention weights using
        # scaled dot product attention
        size = tf.math.sqrt(
            tf.cast(tf.shape(queries)[-1], tf.float32))
        scores = tf.matmul(
            queries, keys, transpose_b=True) / size

        # if an attention bias is provided that add the attention bias
        # to the pre softmax scores matrix
        scores = scores + bias

        # apply a causal mask to the soft attention weights
        mask = tf.expand_dims(values_mask, -2)
        if self.causal:
            mask = tf.logical_and(mask, causal_mask(scores))

        # apply a boolean mask to the keys and values
        scores = tf.math.softmax(tf.where(
            mask, scores, tf.fill(tf.shape(scores), -999999.)))

        # mask the output sequence where appropriate
        outputs = tf.matmul(scores, values)
        if return_scores:
            return tf.where(tf.expand_dims(queries_mask, -1),
                            outputs, tf.zeros_like(outputs)), scores
        else:
            return tf.where(tf.expand_dims(queries_mask, -1),
                            outputs, tf.zeros_like(outputs))

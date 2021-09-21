import tensorflow as tf


class SequenceToMat(tf.keras.layers.Layer):

    def __init__(self, input_size):
        """Creates the backbone for the logits of a permutation layer
        converts a sequence to a matrix of permutation logits"""
        super(SequenceToMat, self).__init__()

        self.norm0 = tf.keras.layers.LayerNormalization()
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.weight_s = self.add_weight(
            name='weight_s', trainable=True,
            shape=(1, input_size, input_size),
            initializer=tf.keras.initializers.GlorotUniform())
        self.weight_v = self.add_weight(
            name='weight_v', trainable=True,
            shape=(1, input_size, input_size),
            initializer=tf.keras.initializers.GlorotUniform())

        self.input_size = input_size

    def static_call(self, queries, keys,
                    queries_mask, values_mask,
                    **kwargs):
        """Runs a forward pass on a multi head attention layer
        inputs is an instance of AttentionInput

        Arguments:

        inputs: queries, keys, and values along with masks

        Returns:

        outputs: tf.Tensor
            the result of applying a multi head attention mechanism
            will be shaped [batch_dim, seq_dim, channels]"""

        # apply dropout to the queries keys and values tensor
        # requires all to be like [batch, heads, ]
        queries = self.norm0(queries, **kwargs)
        keys = self.norm1(keys, **kwargs)

        # compute the multi head soft attention weights using
        # scaled dot product attention
        size = tf.cast(tf.shape(queries)[-1], tf.float32)
        log_s = tf.matmul(queries, tf.matmul(
            self.weight_s, keys, transpose_b=True)) / size
        v = tf.matmul(queries, tf.matmul(
            self.weight_v, keys, transpose_b=True)) / size

        # apply a mask to the scores matrix so that only real
        # non terminal elements are permuted out of place
        mask = tf.expand_dims(values_mask, -2)
        mask = tf.logical_and(mask, tf.expand_dims(queries_mask, -1))

        # pad tokens should not be permuted and logits on the diagonal
        # for pad tokens should not be masked out; this is necessary because
        # a valid permutation matrix has rows and columns that sum to one,
        # even for rows that correspond to pad tokens
        eye = tf.eye(tf.shape(mask)[-2], num_columns=tf.shape(mask)[-1],
                     batch_shape=tf.shape(mask)[:-2], dtype=tf.bool)
        diagonal_mask = tf.logical_and(tf.logical_not(mask), eye)

        # apply a boolean mack to the log scale and mean
        log_s = tf.where(mask, log_s, tf.fill(tf.shape(log_s), -999999.))
        v = tf.where(mask, v, tf.fill(tf.shape(v), -999999.))
        return log_s, tf.where(
            diagonal_mask, tf.fill(tf.shape(v), 999999.), v)

    def call(self, inputs, **kwargs):
        """Runs a forward pass on a multi head attention layer
        inputs is an instance of AttentionInput

        Arguments:

        inputs: AttentionInput
            a dataclass instance that contains queries, keys
            and values along with masks

        Returns:

        outputs: tf.Tensor
            the result of applying a multi head attention mechanism
            will be shaped [batch_dim, seq_dim, channels]"""

        return self.static_call(inputs.queries, inputs.keys,
                                inputs.queries_mask, inputs.values_mask,
                                **kwargs)
from voi.nn.wrappers.layer import Layer
from voi.nn.base.block import Block
from voi.nn.base.attention import Attention
from voi.common_enum import *
import tensorflow as tf

class DecoderLayer(Layer):

    def __init__(self,
                 input_size,
                 hidden_size,
                 heads,
                 queries_dropout=0.,
                 keys_dropout=0.,
                 values_dropout=0.,
                 causal=True,
                 **kwargs):
        """Creates a Transformer decoder layer by applying a
        multi head attention layer

        Arguments:

        input_size: int
            the number of units in the input tensor to this layer
            also the output size of the model
        hidden_size: int
            the number of units in the hidden variables used
            in each multi head attention layer
        heads: int
            the number of heads in each multi head attention layer
            a good default is 4 or 8
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
        super(DecoderLayer, self).__init__()

        self.block0 = Block(hidden_size, input_size * 3, **kwargs)
        self.attention0 = Attention(queries_dropout=queries_dropout,
                                    keys_dropout=keys_dropout,
                                    values_dropout=values_dropout,
                                    causal=causal)

        self.block1 = Block(hidden_size, input_size, **kwargs)
        self.block2 = Block(hidden_size, input_size * 2, **kwargs)
        self.attention1 = Attention(queries_dropout=queries_dropout,
                                    keys_dropout=keys_dropout,
                                    values_dropout=values_dropout,
                                    causal=False)
        self.block3 = Block(hidden_size, input_size, **kwargs)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.heads = heads
        self.queries_dropout = queries_dropout
        self.keys_dropout = keys_dropout
        self.values_dropout = values_dropout
        self.causal = causal
        self.kwargs = kwargs

    def call_main(self, inputs, return_scores=False, **kwargs):
        """Runs a forward pass on a multi head attention layer

        Arguments:

        inputs: list of Tensors
        return_scores: bool
            whether to return attention scores            

        Returns:

        outputs: list of Tensors
            the result of applying a multi head attention mechanism
            same shape as inputs"""

        queries = inputs[QUERIES]
        values = inputs[VALUES]
        queries_mask = inputs[QUERIES_MASK]
        values_mask = inputs[VALUES_MASK]

        # calculate the shape of the values tensor before performing attention
        # used when separating the heads from channels
        s0, s1 = tf.shape(queries), tf.shape(values)
        dim = self.input_size // self.heads

        # pass the input through a feed forward processing block and
        # separate heads from channels
        x = self.block0(queries, **kwargs)
        x = tf.transpose(tf.reshape(x, [
            s0[0], s0[1], self.heads, dim * 3]), [0, 2, 1, 3])

        # pass the input through an attention processing block and
        # flatten the heads and channels
        mask0 = tf.expand_dims(queries_mask, 1)
        if not return_scores:
            x = self.attention0([x[..., :dim], x[..., dim:2*dim], x[..., 2*dim:],
                                 mask0, mask0], **kwargs)
        else:
            x, scores0 = self.attention0([x[..., :dim], x[..., dim:2*dim], x[..., 2*dim:],
                                 mask0, mask0], **kwargs)            
        x = tf.reshape(tf.transpose(x, [
            0, 2, 1, 3]), [s0[0], s0[1], self.heads * dim])

        # encoder-decoder cross attention
        queries = queries + x
        y = self.block1(queries, **kwargs)
        y = tf.transpose(tf.reshape(y, [
            s0[0], s0[1], self.heads, dim]), [0, 2, 1, 3])

        x = self.block2(values, **kwargs)
        x = tf.transpose(tf.reshape(x, [
            s1[0], s1[1], self.heads, dim * 2]), [0, 2, 1, 3])

        mask1 = tf.expand_dims(values_mask, 1)
        if not return_scores:
            x = self.attention1([y, x[..., :dim], x[..., dim:],
                                 mask0, mask1], **kwargs)
        else:
            x, scores1 = self.attention1([y, x[..., :dim], x[..., dim:],
                                 mask0, mask1], **kwargs)
        x = tf.reshape(tf.transpose(x, [
            0, 2, 1, 3]), [s0[0], s0[1], self.heads * dim])

        # pass the outputs of the attention through another feed forward
        # processing block a residual connection
        queries = queries + x
        queries = queries + self.block3(queries, **kwargs)
        
        inputs[QUERIES] = queries
        inputs[VALUES] = values
        inputs[QUERIES_MASK] = queries_mask
        inputs[VALUES_MASK] = values_mask        
        return_args = inputs
        if return_scores:
            return_args = (return_args, [scores0, scores1])
        return return_args

    def call(self, inputs, **kwargs):
        return self.call_main(inputs, **kwargs)
    
    def visualize(self, inputs, **kwargs):
        return self.call_main(inputs, return_scores=True, **kwargs)

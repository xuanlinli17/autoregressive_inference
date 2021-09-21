from voi.nn.wrappers.layer import Layer
from voi.nn.base.block import Block
from voi.nn.base.attention_with_bias import AttentionWithBias
from voi.permutation_utils import pt_permutation_to_relative_l2r
from voi.common_enum import *
import tensorflow as tf


class EncoderWithRelativePositionLayer(Layer):

    def __init__(self,
                 input_size,
                 hidden_size,
                 heads,
                 queries_dropout=0.,
                 keys_dropout=0.,
                 values_dropout=0.,
                 causal=True,
                 num_pos=1,
                 **kwargs):
        """Creates a Transformer encoder layer by applying a
        multi head self attention layer, where relative positions 
        are added to the attention bias.

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
            a causal mask to preserve the auto regressive property
        num_pos: int
            number of relative position"""
        super(EncoderWithRelativePositionLayer, self).__init__()

        # the core attention and processing variables
        self.block0 = Block(hidden_size, input_size * 3, **kwargs)
        self.pos_embedding = tf.keras.layers.Dense(input_size, **kwargs)
        self.attention = AttentionWithBias(queries_dropout=queries_dropout,
                                   keys_dropout=keys_dropout,
                                   values_dropout=values_dropout,
                                   causal=causal)
        self.block1 = Block(hidden_size, input_size, **kwargs)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.heads = heads
        self.queries_dropout = queries_dropout
        self.keys_dropout = keys_dropout
        self.values_dropout = values_dropout
        self.causal = causal
        self.num_pos = num_pos
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

        values = inputs[VALUES]
        values_mask = inputs[VALUES_MASK]
        
        # pass the input through a feed forward processing block and
        # separate heads from channels
        shape, dim = tf.shape(values), self.input_size // self.heads
        x = self.block0(values, **kwargs)
        x = tf.transpose(tf.reshape(x, [
            shape[0], shape[1], self.heads, dim * 3]), [0, 2, 1, 3])
        
        rel = pt_permutation_to_relative_l2r(1, shape[1], tf.constant(self.num_pos))
        # add a position-conditioned bias to the attention scores
        # in log-space: https://arxiv.org/pdf/1902.01370.pdf
        pos = self.pos_embedding(rel, **kwargs)
        pos = tf.transpose(tf.reshape(pos, [
            1, shape[1], shape[1], self.heads, dim]), [0, 3, 1, 2, 4])
        bias = tf.squeeze(tf.matmul(
            tf.expand_dims(x[..., :dim], 3), pos, transpose_b=True), 3)  
        
        # pass the input through an attention processing block and
        # flatten the heads and channels
        mask = tf.expand_dims(values_mask, 1)
        if not return_scores:
            x = self.attention([x[..., :dim], x[..., dim:2*dim], x[..., 2*dim:],
                                mask, mask, bias], **kwargs)
        else:
            x, scores = self.attention([x[..., :dim], x[..., dim:2*dim], x[..., 2*dim:],
                                mask, mask, bias], return_scores=True, **kwargs)
        x = tf.reshape(tf.transpose(x, [
            0, 2, 1, 3]), [shape[0], shape[1], self.heads * dim])

        # pass the outputs of the attention through another feed forward
        # processing block a residual connection
        values = values + x
        values = values + self.block1(values, **kwargs)
        
        inputs[VALUES] = values
        inputs[VALUES_MASK] = values_mask              
        return_args = inputs
        if return_scores:
            return_args = (return_args, [scores])
        return return_args

    def call(self, inputs, **kwargs):
        return self.call_main(inputs, **kwargs)
    
    def visualize(self, inputs, **kwargs):
        return self.call_main(inputs, return_scores=True, **kwargs)  
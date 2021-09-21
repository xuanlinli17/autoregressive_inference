from voi.nn.wrappers.layer import Layer
from voi.nn.base.block import Block
from voi.nn.base.attention import Attention
from voi.nn.base.attention_with_bias import AttentionWithBias
from voi.permutation_utils import pt_permutation_to_relative_l2r
from voi.nn.position_encoding import position_encoding_relative
from voi.common_enum import *
import tensorflow as tf

class DecoderWithRelativePositionalAttentionLayer(Layer):

    def __init__(self,
                 input_size,
                 hidden_size,
                 heads,
                 queries_dropout=0.,
                 keys_dropout=0.,
                 values_dropout=0.,
                 causal=True,
                 **kwargs):
        """
        Relative positional attention as in 
        https://arxiv.org/pdf/1901.02860.pdf
        
        This layer is only used in Permutation transformer

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
        super(DecoderWithRelativePositionalAttentionLayer, self).__init__()
        
        self.relative_length = 100
        self.relative_encoding = position_encoding_relative(self.relative_length,
                                                            input_size) # (range(-100,100), input_size)
        # the core attention and processing variables
        self.block0 = Block(hidden_size, input_size, lastfc=False, **kwargs)
        self.attbias0_0 = tf.keras.layers.Dense(heads, activation=None, **kwargs)
        self.attbias0_1 = tf.keras.layers.Dense(heads, activation=None, **kwargs)    
        self.q0 = tf.keras.layers.Dense(input_size, activation=None, **kwargs)
        self.wke0 = tf.keras.layers.Dense(input_size, activation=None, **kwargs)
        self.wkv0 = tf.keras.layers.Dense(input_size, activation=None, **kwargs)
        self.wkr0 = tf.keras.layers.Dense(input_size, activation=None, **kwargs)        
        self.attention0 = AttentionWithBias(queries_dropout=queries_dropout,
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

        # decoder self-attention with relative-positional encodings as in Transformer-XL
        rel = pt_permutation_to_relative_l2r(1, s0[1], tf.constant(self.relative_length)) #one_hot
        rel = tf.matmul(rel, self.relative_encoding) # (1, shape[1], shape[1], input_size)
        
        x = self.block0(queries, **kwargs)
        new_q = self.q0(x) # (b, shape[1], input_size)
        new_ke = self.wke0(x) # (b, shape[1], input_size)
        new_kv = self.wkv0(x) # (b, shape[1], input_size)
        new_kr = self.wkr0(rel) # (1, shape[1], shape[1], input_size)
        bias0_0 = self.attbias0_0(new_ke) # (b, shape[1], heads)
        bias0_1 = self.attbias0_1(new_kr) # (1, shape[1], shape[1], heads)
        
        def transp(u):
            return tf.transpose(tf.reshape(u, [s0[0], s0[1], self.heads, dim]),
                                [0, 2, 1, 3])
        
        new_q = transp(new_q)
        new_ke = transp(new_ke)
        new_kv = transp(new_kv)
        new_kr = tf.transpose(tf.reshape(new_kr, [1, s0[1], s0[1], self.heads, dim]),
                              [0, 3, 1, 2, 4])
        bias0_0 = tf.transpose(bias0_0, [0, 2, 1])[:, :, tf.newaxis, :]
        bias0_1 = tf.transpose(bias0_1, [0, 3, 1, 2])
        biasprod = tf.reduce_sum(new_q[:, :, :, tf.newaxis, :] * new_kr, axis=-1)
        bias = biasprod + bias0_0 + bias0_1

        # pass the input through an attention processing block and
        # flatten the heads and channels
        mask0 = tf.expand_dims(queries_mask, 1)
        if not return_scores:
            x = self.attention0([new_q, new_ke, new_kv,
                                mask0, mask0, bias], **kwargs)                            
        else:
            x, scores = self.attention0([new_q, new_ke, new_kv,
                                mask0, mask0, bias], return_scores=True, **kwargs)
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
                                          mask0, mask1], return_scores=True, **kwargs)
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


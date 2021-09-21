from voi.nn.wrappers.layer import Layer
from voi.nn.position_encoding import position_encoding
from voi.common_enum import *
import tensorflow as tf


class DiscreteFeature(Layer):

    def __init__(self,
                 hidden_size,
                 src_embedding,
                 tgt_embedding,
                 mode='decoder',
                 sinusoid_pos_emb=False,
                 **kwargs):
        """Creates a Transformer embedding layer by applying a
        lookup operation to the queries, for non-captioning datasets
        where the source is a sequence of word tokens

        Arguments:
        
        hidden_size: int
            the number of units in the hidden variables used
            in each multi head attention layer
        src_embedding: tf.keras.layers.Embedding
            embedding of source vocabulary           
        tgt_embedding: tf.keras.layers.Embedding
            embedding of target vocabulary
        sinusoid_pos_emb: bool
            whether to add sinusoid positional embedding
            to the word embedding
        mode: str
            whether the model using this class
            is the autoregressive decoder or 
            the Permutation Transformer
        """
        super(DiscreteFeature, self).__init__()

        self.hidden_size = hidden_size
        self.mode = mode
        self.sinusoid_pos_emb = sinusoid_pos_emb
        self.kwargs = kwargs

        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding

    def call(self, inputs, **kwargs):
        """
        Arguments:

        inputs: list of Tensors
        
        Returns:

        outputs: list of Tensors
            the result of embedding the inputs 
            (and correctly permuting the inputs if training non-monotonic models)"""
        
        queries = inputs[QUERIES]
        values = inputs[VALUES]  

        q_emb = self.tgt_embedding(queries, **kwargs)
        if self.mode == 'decoder':
            # when training the autoregressive decoder, permute the inputs 
            # based on the input permutation (since we are training using teacher forcing)
            q_emb = tf.matmul(inputs[ABSOLUTE_POSITIONS], q_emb)
        if self.sinusoid_pos_emb:
            q_pos = position_encoding(tf.shape(queries)[1], self.hidden_size)
            q_emb = q_pos + q_emb
            
        v_emb = self.src_embedding(values, **kwargs)
        if self.sinusoid_pos_emb:
            v_pos = position_encoding(tf.shape(values)[1], self.hidden_size)
            v_emb = v_pos + v_emb
            
        inputs[QUERIES] = q_emb
        inputs[VALUES] = v_emb
        return inputs
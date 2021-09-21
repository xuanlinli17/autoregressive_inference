from voi.nn.wrappers.layer import Layer
from voi.nn.position_encoding import position_encoding
from voi.common_enum import *
import tensorflow as tf


class PtDiscreteWithTagsFeature(Layer):

    def __init__(self,
                 hidden_size,
                 tags_embedding_size,
                 src_embedding,
                 tgt_embedding,
                 tags_embedding,
                 sinusoid_pos_emb=False,
                 **kwargs):
        """Creates a Transformer embedding layer.

        For Permutation Transformer only. 
        This layer applies MLP on top of the concatenation of word and parts-of-speech
        embeddings in the target sequence to form the final embedding, 
        if the Permutation Transformer is trained
        using parts-of-speech information.

        Arguments:

        hidden_size: int
            the number of units in the hidden variables used
            in each multi head attention layer
        tags_embedding_size: int
            the embedding dimension for parts of speech
        src_embedding: tf.keras.layers.Embedding
            embedding of vocabulary for source sequences       
        tgt_embedding: tf.keras.layers.Embedding
            embedding of vocabulary for target sequences
        tags_embedding: tf.keras.layers.Embedding
            embedding of parts of speech tags
            for the target
        sinusoid_pos_emb: bool
            whether to add sinusoid positional embedding
            to the word embedding
        mode: str
            whether the model using this class
            is the autoregressive decoder or 
            the Permutation Transformer
        """
        super(PtDiscreteWithTagsFeature, self).__init__()

        assert tags_embedding is not None
        self.hidden_size = hidden_size
        self.sinusoid_pos_emb = sinusoid_pos_emb
        self.kwargs = kwargs

        # the core processing variables
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.tags_embedding = tags_embedding
        self.tags_embedding_size = tags_embedding_size
        self.fc = tf.keras.layers.Dense(hidden_size,
                      activation=None,
                      **kwargs)

    def call(self, inputs, **kwargs):
        """
        Arguments:

        inputs: list of Tensors
        
        Returns:

        outputs: list of Tensors
            the result of embedding the inputs"""
        
        queries = inputs[QUERIES]
        values = inputs[VALUES]
        tags = inputs[TAGS]     

        q_emb = self.tgt_embedding(queries, **kwargs)
        q_tags = self.tags_embedding(tags, **kwargs)
        q_emb = tf.concat([q_emb, q_tags], axis=-1)
        if self.sinusoid_pos_emb:
            q_pos = position_encoding(tf.shape(queries)[1], self.hidden_size + self.tags_embedding_size)
            q_emb = q_pos + q_emb
        q_emb = self.fc(q_emb)
        
        v_emb = self.src_embedding(values, **kwargs)
        if self.sinusoid_pos_emb:
            v_pos = position_encoding(tf.shape(values)[1], self.hidden_size)
            v_emb = v_pos + v_emb
        
        inputs[QUERIES] = q_emb
        inputs[VALUES] = v_emb
        return inputs
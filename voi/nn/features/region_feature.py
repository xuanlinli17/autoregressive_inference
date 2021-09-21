from voi.nn.wrappers.layer import Layer
from voi.nn.position_encoding import position_encoding
from voi.common_enum import *
import tensorflow as tf


class RegionFeature(Layer):

    def __init__(self,
                 hidden_size,
                 queries_embedding,
                 values_embedding,
                 mode='decoder',
                 sinusoid_pos_emb=False,
                 **kwargs):
        """Creates a Transformer embedding layer by applying a
        lookup operation to the queries

        Arguments:

        hidden_size: int
            the number of units in the hidden variables used
            in each multi head attention layer
        queries_embedding: tf.keras.layers.Embedding
            the queries embedding possibly shared between the decoder
            and the Permutation Transformer
            in image captioning, this is the source detection
        values_embedding: tf.keras.layers.Embedding
            the values embedding possibly shared between the decoder
            and the Permutation Transformer  
            in image captioning, this is the target caption
        mode: str
            whether the model using this class
            is the autoregressive decoder or 
            the Permutation Transformer
        sinusoid_pos_emb: bool
            whether to add sinusoid positional embedding
            to the word embedding
        """
        super(RegionFeature, self).__init__()

        self.hidden_size = hidden_size
        self.mode = mode
        self.sinusoid_pos_emb = sinusoid_pos_emb
        self.kwargs = kwargs

        self.word_embedding = values_embedding
        self.detection_embedding = queries_embedding
        self.dense = tf.keras.layers.Dense(
            hidden_size, **kwargs)

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
        object_detections = inputs[OBJECT_DETECTIONS]
        object_features = inputs[OBJECT_FEATURES]
        object_boxes = inputs[OBJECT_BOXES]

        y = self.detection_embedding(object_detections, **kwargs)
        values = self.dense(tf.concat([
            object_features, object_boxes, y], 2), **kwargs)
        
        q_emb = self.word_embedding(queries, **kwargs)
        if self.mode == 'decoder':
            # when training the autoregressive decoder, permute the inputs 
            # based on the input permutation (since we are training using teacher forcing)            
            q_emb = tf.matmul(inputs[ABSOLUTE_POSITIONS], q_emb)
        if self.sinusoid_pos_emb:
            q_pos = position_encoding(tf.shape(queries)[1], self.hidden_size)
            q_emb = q_pos + q_emb
            
        inputs[QUERIES] = q_emb
        inputs[VALUES] = values        
        return inputs
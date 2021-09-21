from voi.nn.base.block import Block
from voi.nn.base.attention import causal_mask
from voi.nn.variables.pointer import Pointer
from voi.common_enum import *
import tensorflow as tf


class PointerAfterLogits(Pointer):

    def __init__(self,
                 hidden_size,
                 output_size,
                 logits_size,
                 logits_embedding,
                 causal=True,
                 **kwargs):
        """Creates a pointer network using the first operation
        in the self attention mechanism

        Arguments:

        hidden_size: int
            the number of hidden units in the network blocks
            used by this layer
        output_size: int
            the number of output units used by the network blocks
            used by this layer
        logits_size: int
            the number of units in the vector space of the logits
            of a transformer model
        logits_embedding: tf.keras.layers.Embedding
            the shared embedding matrix for word and pointer 
            prediction               
        causal: bool
            specifies is the transformer should decoding using
            a causal mask to preserve the auto regressive property"""
        super(PointerAfterLogits, self).__init__(
            hidden_size,
            output_size,
            causal=causal,
            **kwargs)

        self.logits_embedding = logits_embedding
        self.logits_size = logits_size

    def call(self, inputs, **kwargs):
        """
        Arguments:

        inputs: list of Tensors

        Returns:

        pointer_logits: tf.Tensor, shape = [batch, len-1, len-1]
            the logits of a pointer network used to select locations to
            insert words in a transformer
        target_left: tf.Tensor, shape = [batch, len-1, len-1]
            target_left[b,i,j] = at decoding step i, the unsorted position
            of the token with sorted position (partial_pos[b,i,j]-1)
            (inserting to the left of <start> means inserting to the end of sequence)
        target_right: tf.Tensor, shape = [batch, len-1, len-1]
            target_right[b,i,j] = at decoding step i, the unsorted position
            of the token with sorted position (partial_pos[b,i,j]+1)
            
            please refer to the "unsorted" and "sorted" definitions in permutation_utils.py
            
            e.g. in the example in permutation_utils.py, 
            partial_pos[b,:,:] == [[0,inf,inf,inf],[0,1,inf,inf],[0,1,2,inf],[0,2,3,1]]
            target_left[b,:,:] == [[0,1,1,1],[1,0,2,2],[2,0,1,3],[2,3,1,0]]
            -> lower triangular [[0],[1,0],[2,0,1],[2,3,1,0]]
            target_right[b,:,:] == [[0,1,1,1],[1,0,2,2],[1,2,0,3],[3,2,0,1]]
            -> lower triangular [[0],[1,0],[1,2,0],[3,2,0,1]]
            (entries above diagonals are meaningless)
        """

        queries = inputs[QUERIES] # shape [batch, len-1, d]
        queries_mask = inputs[QUERIES_MASK] # shape [batch, len-1]
        ids = inputs[IDS]
        permutation = inputs[PERMUTATION]
        partial_pos = inputs[PARTIAL_POS]

        # map the sequence into a latent space
        features = self.block(queries, **kwargs)
        q = features[..., :self.output_size]
        
        # during training of autoregressive decoder, 
        # we need to permute the original sequence (which is in left-to-right) to
        # use the correct token under the decoding order given by the input permutation
        # (i.e. under unsorted positions) and add its embedding to the hidden
        # vector at the corresponding decoding step to calculate the insertion position later;
        # however during inference, the ids are already in unsorted positions, so we don't need to permute
        embedding_before_permutation = self.logits_embedding(ids)
        if permutation != None:
            q = q + tf.matmul(permutation[:, 1:, 1:],
                              embedding_before_permutation)
        else:
            q = q + embedding_before_permutation        

        # reshape keys to have logits_per_slot insertion positions per token
        shape = tf.multiply(tf.shape(q), [1, self.logits_per_slot, 1])
        k = tf.reshape(features[..., self.output_size:], shape)
        size = tf.math.sqrt(tf.cast(tf.shape(q)[2], tf.float32))
        
        valid_range = tf.range(tf.shape(partial_pos)[-1]) + 1
        valid_range = valid_range[tf.newaxis, :, tf.newaxis]
        target_left = tf.math.floormod(partial_pos - 1, valid_range)
        target_right = tf.math.floormod(partial_pos + 1, valid_range)
        mask = tf.linalg.band_part(tf.ones_like(partial_pos), -1, 0)
        mask = tf.cast(mask, tf.bool)
        partial_pos = tf.where(mask, partial_pos, 999999)
        target_left = tf.where(mask, target_left, 999999)
        target_right = tf.where(mask, target_right, 999999)
        target_left = tf.math.argmin(
            tf.abs(partial_pos[:, :, tf.newaxis, :]
                   - target_left[:, :, :, tf.newaxis]
                  ), 
            axis=-1,
            output_type=tf.int32
        )             
        target_right = tf.math.argmin(
            tf.abs(partial_pos[:, :, tf.newaxis, :]
                   - target_right[:, :, :, tf.newaxis]
                  ), 
            axis=-1,
            output_type=tf.int32
        )  
        
        # calculate raw logit for scores
        # logits[:,:,::self.logits_per_slot] is the probability of inserting to
        # a token's left, while logits[:,:,1::self.logits_per_slot] is the probability
        # of inserting to a token's right
        raw_logits = tf.matmul(q, k, transpose_b=True) / size
        #tf.print("raw_logits", tf.transpose(raw_logits[0][:,:6], [0,1]), summarize=-1)
        
        # prevent the permutation matrix from assigning mass to
        # out of bounds elements
        mask = tf.logical_and(tf.expand_dims(queries_mask, 2),
                              tf.expand_dims(queries_mask, 1))
        if self.causal:
            cmsk = causal_mask(raw_logits[:, tf.newaxis, :, ::self.logits_per_slot])
            mask = tf.logical_and(mask, tf.squeeze(cmsk, 1))

        # filter by removing logits for elements that are invalid
        # mask must be repeated to correct the shape
        mask = tf.repeat(mask, self.logits_per_slot, axis=2)
        return (
            tf.where(mask, raw_logits, 
                     tf.fill(tf.shape(raw_logits), -999999.)
                    ),
            target_left, 
            target_right
        )


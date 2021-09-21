from voi.nn.wrappers.layer import Layer
from voi.nn.base.block import Block
from voi.nn.base.attention import causal_mask
from voi.common_enum import *
from voi.misc import custom_gather
import tensorflow as tf
import tensorflow_probability as tfp
import tree     

@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32)])
def partial_pos_from_hard_relative(R):
    """ Calculates the partial position matrix given the relative position matrix;
        this function is only called during greedy/beam search at inference
        
        Arguments:
        
            R: shape [batch, len-1, len-1, 3]
        
        Returns:
            
            partial_pos: shape [batch, len-1, len-1]
    """
    #assert len(tf.shape(R)) == 4
    R = tf.argmax(R, axis=-1, output_type=tf.int32) - 1
    partial_pos = tf.repeat(
        R[:, tf.newaxis, :, :], tf.shape(R)[-1], axis=-3)
    partial_pos = tf.linalg.band_part(
        tf.transpose(partial_pos, [0, 3, 2, 1]), 0, -1)
    partial_pos = tf.linalg.band_part(
        tf.transpose(partial_pos, [0, 2, 1, 3]), 0, -1)
    partial_pos = tf.cast(
        tf.reduce_sum(
            tf.maximum(
                0, 
                tf.transpose(partial_pos, [0, 3, 1, 2])
            ), 
            axis=-2
        ), 
        tf.int32
    )
    return partial_pos 

@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
    tf.TensorSpec(shape=[None, None, None], dtype=tf.int32),
    tf.TensorSpec(shape=[None, None, None], dtype=tf.int32)])
def prob_both_insertion(raw_logits, 
                        target_left, target_right):
    """ Returns the total probability of inserting at a token's right,
        by normalizing the case that inserting to both left and right is allowed;
        specifically, at a decoding step, given a token with sorted position x,
        we add the probability of inserting at this token's left
        to the probability of inserting at the 
        (token with sorted position x-1)'s right; 
        the returned value is passed through
        cross entropy with inputs.right_pointer_labels to calculate loss
    """                          
    probs = tf.math.softmax(raw_logits, axis=-1)   
    probs_left = probs[..., ::2]
    probs_right = probs[..., 1::2]
    probs_left = custom_gather(probs_left, target_right)
#         tf.print("probs left", probs_left[0,0], summarize=-1)
#         tf.print("probs right", probs_right[0,0], summarize=-1)
    probs = probs_left + probs_right     
    return probs


class Pointer(Layer):

    def __init__(self,
                 hidden_size,
                 output_size,
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
        causal: bool
            specifies is the transformer should decoding using
            a causal mask to preserve the auto regressive propertyt"""
        super(Pointer, self).__init__()

        """
        logits_per_slot: int
            specifies the number of logits per element the pointer
            network attends to
            1: the model inserts only to a token's right (unimplemented)
            2: the model can insert to both a token's left and a token's righ
        """
        self.logits_per_slot = 2
        
        # the core processing variables
        self.block = Block(
            hidden_size, output_size * (1 + self.logits_per_slot), **kwargs)

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.causal = causal
        self.kwargs = kwargs
    
    def call(self, inputs, **kwargs):
        """Using the call function of the subclass that inherits Pointer
        is required:
        e.g. we use "call" in pointer_after_logits.py """

        raise NotImplementedError

    def loss(self, inputs, **kwargs):
        """
        Arguments:

        inputs: list of Tensors

        Returns:

        loss: tf.Tensor
            a loss function that computes the contribution this layer
            makes to the total model loss
        inputs: list of tf.Tensor
            contains the logits of a transformer model used for
            a pointer network"""
        
        pointer_labels = inputs[POINTER_LABELS]
        
        if self.logits_per_slot > 2:
            raise NotImplementedError("logits per slots > 2 not implemented yet")
            
        pointer, target_left, target_right = self.call(inputs, **kwargs)
        probs = prob_both_insertion(pointer, target_left, target_right)
        loss = tf.keras.losses.categorical_crossentropy(
                   pointer_labels, probs, from_logits=False)
        pointer_probs = -loss
        inputs[POINTER_PROBS] = pointer_probs
        return loss, inputs

    def greedy_search(self,
                      inputs,
                      closed,
                      **kwargs):
        """A function that implements a forward pass and updates the decoding
        partial sequence using greedy search

        Arguments:

        inputs: list of Tensors
        closed: tf.Tensor
            a boolean tensor where true values indicate that a beam has
            finished decoding and should not be modified

        Returns:

        decoding: list of Tensors
        closed: tf.Tensor
            a boolean tensor where true values indicate that a beam has
            finished decoding and should not be modified"""

        queries = inputs[QUERIES]
        ids = inputs[IDS]
        relative_positions = inputs[RELATIVE_POSITIONS]
        partial_pos = inputs[PARTIAL_POS]
        log_probs = inputs[LOG_PROBS]
        
        # compute a distribution over tokens; note that only one token
        # is sampled yet top_k is a convenient function        
        raw_logits, target_left, target_right = self.call(inputs, **kwargs)
        probs = prob_both_insertion(raw_logits, target_left, target_right)
        logits = tf.math.log(probs[:, -1, :] + 1e-7)            
        _log_probs, _ids = tf.math.top_k(logits, k=1)        

        # calculate the position of the rightmost token
        R = relative_positions
        R = tf.argmax(R, axis=-1, output_type=tf.int32) - 1
        rightmost_ids = tf.argmax(tf.reduce_sum(
            tf.nn.relu(R), axis=-2), axis=-1, output_type=tf.int32)

        # mask the log probabilities and tokens of already completed
        # beams so that they are unchanged when decoding
        mask = closed[:, tf.newaxis]
        _log_probs = tf.where(mask, tf.zeros_like(_log_probs), _log_probs)
        _ids = tf.where(mask, rightmost_ids[:, tf.newaxis], _ids)

        # compute the relative position update vector using the samples ids
        # this equals -1 if ids are to the left and +1 if to the right
        r = tf.gather(R, _ids, batch_dims=1, axis=2)
        r = tf.squeeze(tf.where(tf.equal(r, 0), tf.ones_like(r), r), axis=2)

        # concatenate the relative position vector to the left and to the
        # bottom of the relative position matrix; see the paper
        # https://arxiv.org/pdf/1902.01370.pdf
        relative_positions = tf.one_hot(tf.concat([
            tf.concat([R, r[:, :, tf.newaxis]], axis=2),
            tf.pad(-r, [[0, 0], [0, 1]])[:, tf.newaxis, :]], axis=1) + 1, 3)
        partial_pos = partial_pos_from_hard_relative(relative_positions)

        # compute the update log probability and note that the pointer network
        # does not specify a termination condition by itself
        log_probs = log_probs + _log_probs[..., 0]
        
        inputs[QUERIES] = queries
        inputs[IDS] = ids
        inputs[RELATIVE_POSITIONS] = relative_positions
        inputs[PARTIAL_POS] = partial_pos
        inputs[LOG_PROBS] = log_probs            
        return (inputs, closed)

    def nucleus_sampling(self,
                         inputs,
                         closed,
                         nucleus_probability,
                         **kwargs):
        """A function that implements a forward pass and updates the decoding
        partial sequence using the nucleas sampling strategy

        Arguments:

        inputs: list of Tensors
        closed: tf.Tensor
            a boolean tensor where true values indicate that a beam has
            finished decoding and should not be modified
        nucleus_probability: float
            the probability threshold used to determine the size
            of the nucleus set of tokens to sample from

        Returns:

        decoding: list of Tensors
        closed: tf.Tensor
            a boolean tensor where true values indicate that a beam has
            finished decoding and should not be modified"""

        queries = inputs[QUERIES]
        ids = inputs[IDS]
        relative_positions = inputs[RELATIVE_POSITIONS]
        partial_pos = inputs[PARTIAL_POS]
        log_probs = inputs[LOG_PROBS]

        # compute a distribution over tokens; note that only one token
        # is sampled yet top_k is a convenient function
        raw_logits, target_left, target_right = self.call(inputs, **kwargs)
        probs = prob_both_insertion(raw_logits, target_left, target_right)
        logits = tf.math.log(probs[:, -1, :] + 1e-7)

        # build an objective to determine the optimal value of k
        sorted_logits = tf.sort(logits, axis=-1, direction='DESCENDING')
        set_probs = tf.math.cumsum(tf.exp(sorted_logits), axis=-1)
        objective = tf.abs(set_probs - nucleus_probability) + \
                    999999.0 * tf.cast(
            set_probs < nucleus_probability, tf.float32)

        # an adaptive value of k that is different for different
        # elements in a batch, for nucleas sampling
        k = tf.math.argmin(objective, axis=-1, output_type=tf.int32)
        min_logits = tf.gather(sorted_logits, k[..., tf.newaxis], batch_dims=1)

        # build a distribution to sample from using the selected k
        # we take the maximum k so that each batch can
        # be processed at the same time
        _log_probs, _ids = tf.math.top_k(logits, k=tf.reduce_max(k) + 1)

        # mask the log probs to remove tokens with low probability
        # then normalize the distribution by subtracting the log denominator
        _log_probs -= 999999.0 * tf.cast(_log_probs < min_logits, tf.float32)
        _log_probs = tf.math.log_softmax(_log_probs)

        # sample from the probability distribution represented
        # by the nucleas set of tokens
        dist = tfp.distributions.Categorical(logits=_log_probs)
        nucleus_samples = dist.sample()[..., tf.newaxis]

        # aggregate the samples according to which token from
        # the nucleas was selected
        _log_probs = tf.gather(_log_probs, nucleus_samples, batch_dims=1)
        _ids = tf.gather(_ids, nucleus_samples, batch_dims=1)

        # calculate the position of the rightmost token
        R = relative_positions
        R = tf.argmax(R, axis=-1, output_type=tf.int32) - 1
        rightmost_ids = tf.argmax(tf.reduce_sum(
            tf.nn.relu(R), axis=-2), axis=-1, output_type=tf.int32)

        # mask the log probabilities and tokens of already completed
        # beams so that they are unchanged when decoding
        mask = closed[:, tf.newaxis]
        _log_probs = tf.where(mask, tf.zeros_like(_log_probs), _log_probs)
        _ids = tf.where(mask, rightmost_ids[:, tf.newaxis], _ids)

        # compute the relative position update vector using the samples ids
        # this equals -1 if ids are to the left and +1 if to the right
        r = tf.gather(R, _ids, batch_dims=1, axis=2)
        r = tf.squeeze(tf.where(tf.equal(r, 0), tf.ones_like(r), r), axis=2)

        # concatenate the relative position vector to the left and to the
        # bottom of the relative position matrix; see the paper
        # https://arxiv.org/pdf/1902.01370.pdf
        relative_positions = tf.one_hot(tf.concat([
            tf.concat([R, r[:, :, tf.newaxis]], axis=2),
            tf.pad(-r, [[0, 0], [0, 1]])[:, tf.newaxis, :]], axis=1) + 1, 3)
        partial_pos = partial_pos_from_hard_relative(relative_positions)

        # compute the update log probability and note that the pointer network
        # does not specify a termination condition by itself
        log_probs = log_probs + _log_probs[..., 0]
        
        inputs[QUERIES] = queries
        inputs[IDS] = ids
        inputs[RELATIVE_POSITIONS] = relative_positions
        inputs[PARTIAL_POS] = partial_pos
        inputs[LOG_PROBS] = log_probs           
        return (inputs, closed)

    def beam_search(self,
                    inputs,
                    closed,
                    last_beam_size,
                    beam_size,
                    **kwargs):
        """A function that implements a forward pass and updates the decoding
        partial sequence using a beam search

        Arguments:

        inputs: list of Tensors
        closed: tf.Tensor
            a boolean tensor where true values indicate that a beam has
            finished decoding and should not be modified
        last_beam_size: int
            the number of beams that were expanded by the last layer in an
            autoregressive model
        beam_size: int
            the number of beams to be expanded by this layer in an
            autoregressive model

        Returns:

        decoding: list of Tensors
        closed: tf.Tensor
            a boolean tensor where true values indicate that a beam has
            finished decoding and should not be modified
        beam_size: int
            the number of beams to be expanded by this layer in an
            autoregressive model"""

        queries = inputs[QUERIES]
        values = inputs[VALUES]
        queries_mask = inputs[QUERIES_MASK]
        values_mask = inputs[VALUES_MASK]
        ids = inputs[IDS]
        permutation = inputs[PERMUTATION]
        absolute_positions = inputs[ABSOLUTE_POSITIONS]
        relative_positions = inputs[RELATIVE_POSITIONS]
        pointer_labels = inputs[POINTER_LABELS]
        logits_labels = inputs[LOGITS_LABELS]
        partial_pos = inputs[PARTIAL_POS]
        pointer_probs = inputs[POINTER_PROBS]
        log_probs = inputs[LOG_PROBS]
        object_detections = inputs[OBJECT_DETECTIONS]
        object_features = inputs[OBJECT_FEATURES]
        object_boxes = inputs[OBJECT_BOXES]
        
        # compute a distribution over pointer locations
        raw_logits, target_left, target_right = self.call(inputs, **kwargs)
        probs = prob_both_insertion(raw_logits, target_left, target_right)
        logits = tf.math.log(probs[:, -1, :] + 1e-7)
        
        batch_size = tf.shape(logits)[0] // last_beam_size

        # note that when the sequence length is small the number of locations
        # that are visible to the pointer network may be too small; the
        # effective beam size is reduced in these cases
        sample_size = tf.minimum(tf.shape(logits)[1], beam_size)

        # sample the top beam_size candidates
        _log_probs, _ids = tf.math.top_k(logits, k=sample_size)

        # when a beam is closed all candidates are the same
        # this prevents the same candidates from being sampled twice
        first = tf.one_hot(tf.fill(tf.shape(_log_probs)[:1], 0), sample_size)
        closed_log_probs = tf.where(tf.equal(first, 0), tf.fill(
            tf.shape(first), -999999.), tf.fill(tf.shape(first), 0.))

        # calculate the position of the rightmost token
        R = relative_positions
        R = tf.argmax(R, axis=-1, output_type=tf.int32) - 1
        rightmost_ids = tf.argmax(tf.reduce_sum(
            tf.nn.relu(R), axis=-2), axis=-1, output_type=tf.int32)

        # when a beam is closed special behavior is required
        # do not change the log probability and append only pad tokens
        mask = closed[:, tf.newaxis]
        _log_probs = tf.where(mask, closed_log_probs, _log_probs)
        _ids = tf.where(mask, rightmost_ids[:, tf.newaxis], _ids)

        # manipulate the log probabilities to extract all possible
        # next beam candidates and their probability
        _log_probs = tf.reshape(_log_probs, [
            batch_size, last_beam_size, sample_size])
        _log_probs = tf.reshape(log_probs, [
            batch_size, last_beam_size, 1]) + _log_probs
        _log_probs = tf.reshape(_log_probs, [
            batch_size, last_beam_size * sample_size])

        # note that when the sequence length is small the number of locations
        # that are visible to the pointer network may be too small; the
        # effective beam size is reduced in these cases
        cand_size = tf.minimum(tf.shape(_log_probs)[1], beam_size)

        # select the top beam_size candidates
        _log_probs, beam_ids = tf.math.top_k(_log_probs, k=cand_size)

        # these indices may be a bit subtle; they work as follows
        # the last dim has last_beam_size * beam_size elements
        # the first beam_size elements represent candidate proposals
        # from a single original beam
        old_beam_ids = tf.math.floordiv(beam_ids, sample_size)

        # select the ids based on their beams that are from the beams with
        # highest log probability
        _ids = tf.reshape(_ids, [batch_size, last_beam_size * sample_size])
        _ids = tf.gather(_ids, beam_ids, batch_dims=1)
        _ids = tf.reshape(_ids, [batch_size * cand_size, 1])

        # this function helps select the hidden activations from
        # inputs that correspond to old selected beams
        # this is necessary because future layers may depend on activations
        # that are a function of which beam was selected
        def select(x):
            if x is None:
                return x
            shape = tf.shape(x)[1:]
            s0 = tf.concat([[batch_size, last_beam_size], shape], axis=0)
            s1 = tf.concat([[batch_size * cand_size], shape], axis=0)
            return tf.reshape(tf.gather(
                tf.reshape(x, s0), old_beam_ids, batch_dims=1), s1)

        # select which old beams are propagated forward
        # this is necessary because some beams have content-aware state
        queries = select(queries)
        values = select(values)
        queries_mask = select(queries_mask)
        values_mask = select(values_mask)
        ids = select(ids)
        permutation = select(permutation)
        absolute_positions = select(absolute_positions)
        relative_positions = select(relative_positions)
        partial_pos = select(partial_pos)
        pointer_labels = select(pointer_labels)
        logits_labels = select(logits_labels)

        # TODO: Brandon -> handle the image features as well.
        object_detections = select(object_detections)
        object_features = select(object_features)
        object_boxes = select(object_boxes)

        # compute the relative position update vector using the samples ids
        # this equals -1 if ids are to the left and +1 if to the right
        R = relative_positions
        R = tf.argmax(R, axis=-1, output_type=tf.int32) - 1
        r = tf.gather(R, _ids, batch_dims=1, axis=2)
        r = tf.squeeze(tf.where(tf.equal(r, 0), tf.ones_like(r), r), axis=2)

        # concatenate the relative position vector to the left and to the
        # bottom of the relative position matrix; see the paper
        # https://arxiv.org/pdf/1902.01370.pdf
        relative_positions = tf.one_hot(tf.concat([
            tf.concat([R, r[:, :, tf.newaxis]], axis=2),
            tf.pad(-r, [[0, 0], [0, 1]])[:, tf.newaxis, :]], axis=1) + 1, 3)
        partial_pos = partial_pos_from_hard_relative(relative_positions)

        # update log probability and note that the pointer network
        # does not specify a termination condition by itself
        log_probs = tf.reshape(_log_probs, [batch_size * cand_size])
        
        inputs[QUERIES] = queries
        inputs[VALUES] = values
        inputs[QUERIES_MASK] = queries_mask
        inputs[VALUES_MASK] = values_mask
        inputs[IDS] = ids
        inputs[PERMUTATION] = permutation
        inputs[ABSOLUTE_POSITIONS] = absolute_positions
        inputs[RELATIVE_POSITIONS] = relative_positions
        inputs[POINTER_LABELS] = pointer_labels
        inputs[LOGITS_LABELS] = logits_labels
        inputs[PARTIAL_POS] = partial_pos
        inputs[POINTER_PROBS] = pointer_probs
        inputs[LOG_PROBS] = log_probs    
        inputs[OBJECT_DETECTIONS] = object_detections
        inputs[OBJECT_FEATURES] = object_features
        inputs[OBJECT_BOXES] = object_boxes        
        return (inputs, select(closed), cand_size)

    def adaptive_search(self,
                        inputs,
                        closed,
                        last_beam_size,
                        beam_size,
                        natural_order_tokens,
                        natural_order_pos,
                        **kwargs):
        """A function that implements a forward pass and updates the decoding
        partial sequence using a beam search

        Arguments:

        inputs: list of Tensors
        closed: tf.Tensor
            a boolean tensor where true values indicate that a beam has
            finished decoding and should not be modified
        last_beam_size: int
            the number of beams that were expanded by the last layer in an
            autoregressive model
        beam_size: int
            the number of beams to be expanded by this layer in an
            autoregressive model
        natural_order_tokens: tf.Tensor
            a batch of sequences representing the generation index of tokens
            in natural order that are yet to be decoded.
        natural_order_pos: tf.Tensor
            a batch of sequences representing the word ids of tokens
            in natural order that are yet to be decoded.

        Returns:

        updated variables above"""

        queries = inputs[QUERIES]
        values = inputs[VALUES]
        queries_mask = inputs[QUERIES_MASK]
        values_mask = inputs[VALUES_MASK]
        ids = inputs[IDS]
        permutation = inputs[PERMUTATION]
        absolute_positions = inputs[ABSOLUTE_POSITIONS]
        relative_positions = inputs[RELATIVE_POSITIONS]
        pointer_labels = inputs[POINTER_LABELS]
        logits_labels = inputs[LOGITS_LABELS]
        partial_pos = inputs[PARTIAL_POS]
        pointer_probs = inputs[POINTER_PROBS]
        log_probs = inputs[LOG_PROBS]
        object_detections = inputs[OBJECT_DETECTIONS]
        object_features = inputs[OBJECT_FEATURES]
        object_boxes = inputs[OBJECT_BOXES]

        # compute the possible absolute locations of that word
        # shaped like: [batch_size x candidate_slots]
        pos_match = tf.cast(tf.equal(
            natural_order_tokens, ids[:, -1:]), tf.float32)

        # create a mask of all the leftmost words for every absolute position
        # is a one at the location of a word that is to the left
        # NOTE: natural_order_pos corresponds to <start>, ...
        # NOTE: whereas natural_order_tokens corresponds to ..., <end>
        # NOTE: so tf.math.cumsum exclusive should = False
        _pos_mask = tf.eye(tf.shape(natural_order_pos)[1], batch_shape=[1])
        _pos_mask = tf.math.cumsum(_pos_mask, axis=2, reverse=True)

        # build a mask over which leftmost tokens that are generated
        # handles the <start> token which is always inserted at the front
        # because the 0th element of natural_order_pos is always zero
        _gen_mask = tf.cast(tf.greater_equal(natural_order_pos, 0), tf.float32)
        _gen_mask = _gen_mask[:, tf.newaxis, :] * _pos_mask

        # filter for the rightmost left tokens
        _rank = _gen_mask * tf.math.cumsum(_gen_mask, axis=2)
        _max_rank = tf.reduce_max(_rank, axis=2, keepdims=True)
        _gen_mask = tf.cast(tf.equal(_rank, _max_rank), tf.float32)

        # at this point _gen_mask is a tensor
        # shaped like: [batch_size x candidate_slots x absolute_length]
        # and contains a 1 at the rightmost left token position
        # unless the rlt is <start>, in which case the tensor is zero.
        # convert it to [batch_size x absolute_length]
        rlt_mask = tf.clip_by_value(tf.reduce_sum(
            pos_match[..., tf.newaxis] * _gen_mask, axis=1), 0.0, 1.0)

        # compute a mask over generation indices shaped like:
        #   [batch_size x absolute_length x generation_indices]
        all_gen_pos = tf.one_hot(
            natural_order_pos, tf.shape(queries)[1], axis=2)

        # sum over the number of candidate pointer locations: candidate_slots
        # and over the selector for rlt: absolute_length
        pos_mask = tf.reduce_sum(
            rlt_mask[..., tf.newaxis] * all_gen_pos, axis=1)

        # this mask contains in every possible location
        # that the previously decoded word can be generated at
        pos_mask = tf.clip_by_value(pos_mask, 0.0, 1.0)

        # compute a distribution over pointer locations
        logits, target_left, target_right = self.call(inputs, **kwargs)
        probs = prob_both_insertion(logits, target_left, target_right)
        logits = tf.math.log(probs[:, -1, :] + 1e-7)

        # convert the masks into offsets for the softmax op: 0 -> -\infty
        # shaped like: [batch_size x generation_indices]
        offset = (1.0 - pos_mask) * 999999.0
        logits = tf.math.log_softmax(logits - offset)
        batch_size = tf.shape(logits)[0] // last_beam_size

        # note that when the sequence length is small the number of locations
        # that are visible to the pointer network may be too small; the
        # effective beam size is reduced in these cases
        sample_size = tf.minimum(tf.shape(logits)[1], beam_size)

        # sample the top beam_size candidates
        _log_probs, _ids = tf.math.top_k(logits, k=sample_size)

        # when a beam is closed all candidates are the same
        # this prevents the same candidates from being sampled twice
        first = tf.one_hot(tf.fill(tf.shape(_log_probs)[:1], 0), sample_size)
        closed_log_probs = tf.where(tf.equal(first, 0), tf.fill(
            tf.shape(first), -999999.), tf.fill(tf.shape(first), 0.))

        # calculate the position of the rightmost token
        R = relative_positions
        R = tf.argmax(R, axis=-1, output_type=tf.int32) - 1
        rightmost_ids = tf.argmax(tf.reduce_sum(
            tf.nn.relu(R), axis=-2), axis=-1, output_type=tf.int32)

        # when a beam is closed special behavior is required
        # do not change the log probability and append only pad tokens
        mask = closed[:, tf.newaxis]
        _log_probs = tf.where(mask, closed_log_probs, _log_probs)
        _ids = tf.where(mask, rightmost_ids[:, tf.newaxis], _ids)

        # manipulate the log probabilities to extract all possible
        # next beam candidates and their probability
        _log_probs = tf.reshape(_log_probs, [
            batch_size, last_beam_size, sample_size])
        _log_probs = tf.reshape(log_probs, [
            batch_size, last_beam_size, 1]) + _log_probs
        _log_probs = tf.reshape(_log_probs, [
            batch_size, last_beam_size * sample_size])

        # note that when the sequence length is small the number of locations
        # that are visible to the pointer network may be too small; the
        # effective beam size is reduced in these cases
        cand_size = tf.minimum(tf.shape(_log_probs)[1], beam_size)

        # select the top beam_size candidates
        _log_probs, beam_ids = tf.math.top_k(_log_probs, k=cand_size)

        # these indices may be a bit subtle; they work as follows
        # the last dim has last_beam_size * beam_size elements
        # the first beam_size elements represent candidate proposals
        # from a single original beam
        old_beam_ids = tf.math.floordiv(beam_ids, sample_size)

        # select the ids based on their beams that are from the beams with
        # highest log probability
        _ids = tf.reshape(_ids, [batch_size, last_beam_size * sample_size])
        _ids = tf.gather(_ids, beam_ids, batch_dims=1)
        _ids = tf.reshape(_ids, [batch_size * cand_size, 1])

        # this function helps select the hidden activations from
        # inputs that correspond to old selected beams
        # this is necessary because future layers may depend on activations
        # that are a function of which beam was selected
        def select(x):
            if x is None:
                return x
            shape = tf.shape(x)[1:]
            s0 = tf.concat([[batch_size, last_beam_size], shape], axis=0)
            s1 = tf.concat([[batch_size * cand_size], shape], axis=0)
            return tf.reshape(tf.gather(
                tf.reshape(x, s0), old_beam_ids, batch_dims=1), s1)

        # select which old beams are propagated forward
        # this is necessary because some beams have content-aware state
        queries = select(queries)
        values = select(values)
        queries_mask = select(queries_mask)
        values_mask = select(values_mask)
        ids = select(ids)
        permutation = select(permutation)
        absolute_positions = select(absolute_positions)
        relative_positions = select(relative_positions)
        partial_pos = select(partial_pos)
        pointer_labels = select(pointer_labels)
        logits_labels = select(logits_labels)
        natural_order_tokens = select(natural_order_tokens)
        natural_order_pos = select(natural_order_pos)

        # TODO: Brandon -> handle the image features as well.
        object_detections = select(object_detections)
        object_features = select(object_features)
        object_boxes = select(object_boxes)

        # compute the relative position update vector using the samples ids
        # this equals -1 if ids are to the left and +1 if to the right
        R = relative_positions
        R = tf.argmax(R, axis=-1, output_type=tf.int32) - 1
        r = tf.gather(R, _ids, batch_dims=1, axis=2)
        r = tf.squeeze(tf.where(tf.equal(r, 0), tf.ones_like(r), r), axis=2)

        # concatenate the relative position vector to the left and to the
        # bottom of the relative position matrix; see the paper
        # https://arxiv.org/pdf/1902.01370.pdf
        relative_positions = tf.one_hot(tf.concat([
            tf.concat([R, r[:, :, tf.newaxis]], axis=2),
            tf.pad(-r, [[0, 0], [0, 1]])[:, tf.newaxis, :]], axis=1) + 1, 3)
        partial_pos = partial_pos_from_hard_relative(relative_positions)

        # update the natural order sentence to do two things:
        #   (1) make each word that is already decoded a <pad> token
        #   (2) make each position that is decoded equal to its generation index

        rlt_pos = tf.math.argmin(tf.math.abs(
            natural_order_pos - _ids), axis=1, output_type=tf.int32)

        can_pos = tf.logical_and(
            tf.equal(natural_order_tokens, ids[:, -1:]),
            tf.less_equal(rlt_pos[:, tf.newaxis], tf.range(
                tf.shape(natural_order_tokens)[1])[tf.newaxis]))

        # there can be several matches, so pick the first
        # other choices are possible, such as the last, or randomly
        sel_pos = tf.math.argmax(
            tf.cast(can_pos, tf.float32), axis=1, output_type=tf.int32)
        shift_pos = tf.clip_by_value(
            sel_pos + 1, 0, tf.shape(natural_order_tokens)[1] - 1)

        # choose the positions into natural_order to replace
        indices = tf.stack([tf.range(tf.shape(
            natural_order_tokens)[0]), sel_pos], axis=1)

        shift_indices = tf.stack([tf.range(tf.shape(
            natural_order_tokens)[0]), shift_pos], axis=1)

        # replace this position with <pad> tokens in the natural_order_tokens
        fill_token = tf.fill([tf.shape(natural_order_tokens)[0]], 0)
        natural_order_tokens = tf.tensor_scatter_nd_update(
            natural_order_tokens, indices, fill_token)

        # replace this position with len(ids) positions in the natural_order_pos
        # at the last iteration, this will write to an invalid location,
        # but this behavior is fine
        fill_pos = tf.fill([
            tf.shape(natural_order_tokens)[0]], tf.shape(queries)[1])
        natural_order_pos = tf.tensor_scatter_nd_update(
            natural_order_pos, shift_indices, fill_pos)

        # update log probability and note that the pointer network
        # does not specify a termination condition by itself
        log_probs = tf.reshape(_log_probs, [batch_size * cand_size])
        
        inputs[QUERIES] = queries
        inputs[VALUES] = values
        inputs[QUERIES_MASK] = queries_mask
        inputs[VALUES_MASK] = values_mask
        inputs[IDS] = ids
        inputs[PERMUTATION] = permutation
        inputs[ABSOLUTE_POSITIONS] = absolute_positions
        inputs[RELATIVE_POSITIONS] = relative_positions
        inputs[POINTER_LABELS] = pointer_labels
        inputs[LOGITS_LABELS] = logits_labels
        inputs[PARTIAL_POS] = partial_pos
        inputs[POINTER_PROBS] = pointer_probs
        inputs[LOG_PROBS] = log_probs    
        inputs[OBJECT_DETECTIONS] = object_detections
        inputs[OBJECT_FEATURES] = object_features
        inputs[OBJECT_BOXES] = object_boxes                
        return (inputs, select(closed), cand_size,
                natural_order_tokens, natural_order_pos)
import tensorflow as tf
from voi.common_enum import *

def nucleus_sampling(inputs,
                     model,
                     dataset,
                     max_iterations=20,
                     nucleus_probability=0.95,
                     num_samples=5,
                     return_rel_pos=False):
    """Decode a sequence of tokens using the nucleus sampling
    strategy, and return several independent samples

    Arguments:

    inputs: list of Tensors
    model: Transformer
        the autoregressive decoder
    dataset: str
        the type of dataset
    max_iterations: int
        the maximum number of decoding steps to use when performing
        greedy search; the maximum sequence length
    nucleus_probability: float
        the probability threshold used to determine the size
        of the nucleus set of tokens to sample from
    num_samples: int
        the number of independent identically distributed samples
        to draw from the probability distribution given by the nucleus
    return_rel_pos: bool
        whether to return relative position matrix (for inspect_order.py)

    Returns:

    sequence: tf.Tensor
        a tensor that contains output word ids that were taken
        when decoding using the transformer
    log_p: tf.Tensor
        the log probability of predicted sentences under the
        current transformer model"""

    # unpack all the requires model inputs:
    queries = inputs[QUERIES]
    values = inputs[VALUES]
    queries_mask = inputs[QUERIES_MASK]
    values_mask = inputs[VALUES_MASK]
    ids = inputs[IDS]
    permutation = inputs[PERMUTATION]
    absolute_positions = inputs[ABSOLUTE_POSITIONS]
    relative_positions = inputs[RELATIVE_POSITIONS]
    log_probs = inputs[LOG_PROBS]
    object_detections = inputs[OBJECT_DETECTIONS]
    object_features = inputs[OBJECT_FEATURES]
    object_boxes = inputs[OBJECT_BOXES]

    # meta data to keep track of which beams have completed
    # during the decoding step
    batch_size = tf.shape(values_mask)[0]
    closed = tf.fill([batch_size], False)

    # replace the model inputs with an empty sentence that will be
    # appended to during the decoding step
    start = tf.fill([batch_size, 1], START_ID)
    queries = start
    queries_mask = tf.fill([batch_size, 1], True)

    old_values = values
    if dataset == 'captioning':
        # dummy value variable
        values = queries

    ids = tf.fill([batch_size, 0], START_ID)
    partial_pos = tf.zeros([batch_size, 1, 1], dtype=tf.int32)
    relative_positions = tf.one_hot(tf.fill([batch_size, 1, 1], 1), 3) # corresponds to "self" or "0" in relative_positions
    # tf.eye is used because during inference, the decoded tokens are already in unsorted orderings,
    # as relative_positions takes care of the relationship between the actual sorted positions    
    absolute_positions = tf.eye(1, batch_shape=[batch_size])
    permutation = tf.eye(2, batch_shape=[batch_size])
    # we don't need these labels during inference, but we need log_probs
    pointer_labels = tf.zeros([batch_size])
    logits_labels = tf.zeros([batch_size])
    log_probs = tf.zeros([batch_size])
    pointer_probs = tf.zeros([batch_size])

    # duplicate the features by the number of samples to draw from
    # the nucleus sampling distribution: [batch, num_samples]
    queries = tf.repeat(queries, num_samples, axis=0)
    values = tf.repeat(values, num_samples, axis=0)
    queries_mask = tf.repeat(queries_mask, num_samples, axis=0)
    values_mask = tf.repeat(values_mask, num_samples, axis=0)
    ids = tf.repeat(ids, num_samples, axis=0)

    # duplicate the features by the number of samples to draw from
    # the nucleus sampling distribution: [batch, num_samples]
    partial_pos = tf.repeat(partial_pos, num_samples, axis=0)
    pointer_probs = tf.repeat(pointer_probs, num_samples, axis=0)
    log_probs = tf.repeat(log_probs, num_samples, axis=0)

    # duplicate the features by the number of samples to draw from
    # the nucleus sampling distribution: [batch, num_samples]
    relative_positions = tf.repeat(relative_positions, num_samples, axis=0)
    absolute_positions = tf.repeat(absolute_positions, num_samples, axis=0)
    permutation = tf.repeat(permutation, num_samples, axis=0)
    pointer_labels = tf.repeat(pointer_labels, num_samples, axis=0)
    logits_labels = tf.repeat(logits_labels, num_samples, axis=0)

    # duplicate the features by the number of samples to draw from
    # the nucleus sampling distribution: [batch, num_samples]
    object_detections = tf.repeat(object_detections, num_samples, axis=0)
    object_features = tf.repeat(object_features, num_samples, axis=0)
    object_boxes = tf.repeat(object_boxes, num_samples, axis=0)
    closed = tf.repeat(closed, num_samples, axis=0)

    def update(inputs, closed, i):

        # decode using the model for a single pass
        inputs, closed = model.nucleus_sampling(
            inputs, closed, nucleus_probability)

        # we need to replace the transformer inputs at every
        # iteration of decoding
        start = tf.fill([batch_size * num_samples, 1], START_ID)
        inputs[ABSOLUTE_POSITIONS] = tf.eye(2 + i, batch_shape=[batch_size * num_samples])
        inputs[PERMUTATION] = tf.eye(3 + i, batch_shape=[batch_size * num_samples])
        inputs[QUERIES] = tf.concat([start, inputs[IDS]], axis=1)
        inputs[QUERIES_MASK] = tf.concat([
            inputs[QUERIES_MASK],
            tf.logical_not(closed)[:, tf.newaxis]], axis=1)
        if dataset == 'captioning':
            inputs[VALUES] = inputs[QUERIES]
        elif dataset in ['wmt', 'django', 'gigaword']:
            inputs[VALUES] = tf.repeat(old_values, num_samples, axis=0)

        i = i + 1
        closed = tf.logical_or(closed, tf.greater_equal(i, max_iterations))
        return [inputs, closed, i]

    def cond(inputs, closed, i):
        return tf.logical_not(tf.reduce_all(closed))

    # loop for a maximum of max_iterations decoding steps
    i = tf.constant(0)
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
    outputs, closed, i = tf.while_loop(
        cond,
        update,
        [inputs, closed, i],
        shape_invariants=[
            [
                tf.TensorShape([None, None]),
                tf.TensorShape([None, None]),
                tf.TensorShape([None, None]),
                tf.TensorShape([None, None]),
                tf.TensorShape([None, None]),
                tf.TensorShape([None, None, None]),
                tf.TensorShape([None, None, None]),
                tf.TensorShape([None, None, None, 3]),
                tf.TensorShape([None]),
                tf.TensorShape([None]),
                tf.TensorShape([None, None, None]),
                tf.TensorShape([None]),
                tf.TensorShape([None]),
                tf.TensorShape([None, None]),
                tf.TensorShape([None, None, None]),
                tf.TensorShape([None, None, 4])
            ],
            tf.TensorShape([None]),
            i.get_shape()])

    ids = outputs[IDS]
    relative_positions = outputs[RELATIVE_POSITIONS]
    log_probs = outputs[LOG_PROBS]

    # helper function for un flattening the beam size from the batch axis
    def expand(x):
        return tf.reshape(x, tf.concat([[
            batch_size, num_samples], tf.shape(x)[1:]], axis=0))

    # decoding is finished so un flatten the beam dimension
    # returns a shape like [batch_size, beam_size, sequence_length]
    ids = expand(ids)

    # when the model decodes permutation matrices in additions to ids;
    # then sort ids according to the decoded permutation
    if model.final_layer == 'indigo':
        pos = relative_positions
        pos = tf.argmax(pos, axis=-1, output_type=tf.int32) - 1
        pos = tf.reduce_sum(tf.nn.relu(expand(pos[:, 1:, 1:])), axis=2)
        pos = tf.one_hot(pos, tf.shape(pos)[2], dtype=tf.int32)
        ids = tf.squeeze(tf.matmul(tf.expand_dims(ids, 2), pos), 2)

    if not return_rel_pos:
        return ids, tf.reshape(
            log_probs, [batch_size, num_samples])
    else:
        return ids, tf.reshape(
            log_probs, [batch_size, num_samples]), expand(relative_positions)
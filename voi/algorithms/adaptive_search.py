import tensorflow as tf
from voi.common_enum import *

def adaptive_search(inputs,
                    model,
                    dataset,
                    beam_size=8,
                    max_iterations=20,
                    return_rel_pos=False):
    """Run search adaptive order given a batch of sequences in a natural order
    and return the order with highest probability

    Arguments:

    inputs: list of Tensors
    model: Transformer
        the autoregressive decoder
    dataset: str
        the type of dataset
    beam_size: int
        the number of beams to use when calculating a beam search
        a beam size of zero is a greedy search
    max_iterations: int
        the maximum number of decoding steps to use when performing
        beam search; the maximum sequence length
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

    # build the adaptive search tensors
    natural_order_tokens = ids

    # build a generation index tensor with <start> generated first at position 0
    natural_order_pos = tf.ones_like(natural_order_tokens[:, :1])
    natural_order_pos = tf.pad(natural_order_pos, [
        [0, 0], [0, tf.shape(natural_order_tokens)[1] - 1]]) - 1

    # meta data to keep track of which beams have completed
    # during the decoding step
    batch_size = tf.shape(values_mask)[0]
    closed = tf.fill([batch_size], False)
    last_beam_size = tf.constant(1)

    # replace the model inputs with an empty sentence that will be
    # appended to during the decoding step
    queries = tf.fill([batch_size, 1], START_ID)
    queries_mask = tf.fill([batch_size, 1], True)

    old_values = values
    if dataset == 'captioning':
        # dummy value variable
        values = queries

    ids = tf.fill([batch_size, 0], START_ID)
    relative_positions = tf.one_hot(tf.fill([batch_size, 1, 1], 1), 3) # corresponds to "self" or "0" in relative_positions
    # tf.eye is used because during inference, the decoded tokens are already in unsorted orderings,
    # and relative_positions takes care of the relationship between the actual sorted positions    
    absolute_positions = tf.eye(1, batch_shape=[batch_size])
    partial_pos = tf.zeros([batch_size, 1, 1], dtype=tf.int32)
    permutation = tf.eye(2, batch_shape=[batch_size])
    # we don't need these labels during inference, but we need log_probs
    pointer_labels = tf.zeros([batch_size])
    logits_labels = tf.zeros([batch_size])
    log_probs = tf.zeros([batch_size])
    pointer_probs = tf.zeros([batch_size])

    def update(inputs, closed, last_beam_size,
               natural_order_tokens, natural_order_pos, i):

        [inputs, closed, last_beam_size,
            natural_order_tokens, natural_order_pos] = model.adaptive_search(
                inputs, closed, last_beam_size, beam_size,
                natural_order_tokens, natural_order_pos
        )

        # we need to replace the transformer inputs at every
        # iteration of decoding
        start = tf.fill([batch_size * last_beam_size, 1], START_ID)
        inputs[ABSOLUTE_POSITIONS] = tf.eye(2 + i, batch_shape=[batch_size * last_beam_size])
        inputs[PERMUTATION] = tf.eye(3 + i, batch_shape=[batch_size * last_beam_size])
        inputs[QUERIES] = tf.concat([start, inputs[IDS]], axis=1)
        inputs[QUERIES_MASK] = tf.concat([
            inputs[QUERIES_MASK],
            tf.logical_not(closed)[:, tf.newaxis]], axis=1)
        if dataset == 'captioning':
            inputs[VALUES] = inputs[QUERIES]
        elif dataset in ['wmt', 'django', 'gigaword']:
            inputs[VALUES] = tf.repeat(old_values, last_beam_size, axis=0)

        i = i + 1
        closed = tf.logical_or(closed, tf.greater_equal(i, max_iterations))
        return [inputs, closed, last_beam_size,
                natural_order_tokens, natural_order_pos, i]

    def cond(inputs, closed, last_beam_size,
             natural_order_tokens, natural_order_pos, i):
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
    [outputs, closed, last_beam_size, 
     natural_order_tokens, natural_order_pos, i] = tf.while_loop(
        cond,
        update,
        [inputs, closed, last_beam_size,
         natural_order_tokens, natural_order_pos, i],
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
            last_beam_size.get_shape(),
            tf.TensorShape([None, None]),
            tf.TensorShape([None, None]),
            i.get_shape()])

    ids = outputs[IDS]
    relative_positions = outputs[RELATIVE_POSITIONS]
    log_probs = outputs[LOG_PROBS]

    # helper function for un flattening the beam size from the batch axis
    def expand(x):
        return tf.reshape(x, tf.concat([[
            batch_size, last_beam_size], tf.shape(x)[1:]], axis=0))

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
            log_probs, [batch_size, last_beam_size])
    else:
        return ids, tf.reshape(
            log_probs, [batch_size, last_beam_size]), expand(relative_positions)

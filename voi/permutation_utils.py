import tensorflow as tf


@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, None, None], dtype=tf.float32)])
def permutation_to_pointer(permutation):
    """Converts a permutation matrix to the label distribution of
    a pointer network for training an autoregressive language model

    Arguments:

    permutation: tf.Tensor, shape = [batch, len, len]
        a permutation matrix that defines the order in which
        words should be inserted by the language model
        
        the first dimension is the batch dimension
        the last two dimensions contain the 2d permutation matrix
        
        each entry must be 0/1
        
        the top left element of the permutation matrix must be 1
        since <start> token must always be generated first
        
        the bottom right element of the permutation matrix must be 1
        since it corresponds to the <end> or <pad> token
    Returns:

    pointer: tf.Tensor, shape = [batch, len-1, len-1]
        a ternary matrix that contains the target relative positions of words
        where the language model should insert non-monotonically;
        here the target relative positions are with respect to the previously
        generated tokens under the generation order of the permutation
        
    partial_pos: tf.Tensor, shape = [batch, len-1, len-1]
        at each decoding step, the positions of already-decoded tokens
        sorted according to the ground truth target sequence
    """

    # make sure the permutation is an int or the below computation
    # does not make sense
    permutation = tf.cast(permutation, tf.int32)
    n = tf.shape(permutation)[-1]

    """
    An important thing is that regardless of generation order, 
    i.e. generating which token first and which token last,
    we should always still generate the target sequence where
    the relative position between tokens is independent of such generation orders.
    e.g. we should always generate "<start> I love my dogs <end>" regardless of whether we generate
    <start>->love->dogs->my->I-><end> or <start>->I->love->my->dogs-><end>.
    
    From here on, unsorted[b,i,j] refers to the ith token in the generation order under the permutation
    with respect to the jth token in the generation order under the permutation, while 
    sorted[b,i,j] refers to the ith token in the generation order under the permutation with respect to
    the jth token in the "sorted" ordering according to the ground truth target sequence.
    """
    
    """
    This first section will convert the one-hot style indexing to
    a quaternary indexing, where sorted_relative[b,i,j]==-1 
    means the ith token in the generation order under the permutation 
    is to the left of the jth token in the generation order under left-to-right;
    1 means i is to the right of j, and 0 means i==j.
    
    e.g. if permutation[b,:,:] == [[1,0,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,1,0,0,0],[0,0,0,0,1]], n==5,
    i.e. we generate <start>->token 2->token 3->token 1-><end>, in the ground truth sequence of <start> 1 2 3 <end>,
    then sorted_relative[b,:,:] == [[0,-1,-1,-1,-1],[1,1,0,-1,-1],[1,1,1,0,-1],[1,0,-1,-1,-1],[1,1,1,1,0]]
    """
    sorted_relative = -tf.math.cumsum(
        permutation, axis=-1, exclusive=True) + tf.math.cumsum(
            permutation, axis=-1, exclusive=True, reverse=True)

    """
    If unsorted_relative[b,i,j] == -1, 
    then the ith token in the generation order under the permutation
    should insert to the jth token's right, and here jth token refers to the
    jth token in the generation order under the permutation; if == 1 then i inserts to j's left.
    
    Note that unsorted_relative[b,i,i] always == 0, i.e. the diagnoal entries are always 0.
    
    In the above example, unsorted_relative[b,:,:] == [[0,1,1,1,1],[-1,0,1,-1,1],[-1,-1,0,-1,1],[-1,1,1,0,1],[-1,-1,-1,-1,0]].
    """
    unsorted_relative = tf.matmul(
        permutation, sorted_relative, transpose_b=True)

    """
    Get the one hot distribution of pointer labels; should contain
    a sparse lower triangular matrix.
    
    Here sorted_ptr[b,i] == x means that the ith token in the generation order
    under the permutation should insert to the (x-1)th token's right 
    where the xth token refers to the xth token, in the left-to-right (sorted) order,
    of the tokens generated so far, 
    where the tokens generated so far refers to the [0,1,2,...,i]th token in the generation order
    under the permutation.
    
    Note that sorted_ptr[b,0] is not meaningful because this corresponds to the <start> token which is never inserted;
    this entry will be ignored later. Also sorted_ptr[b,i>0] > 0.
    
    In the above example, sorted_ptr[b,:] == [0,1,2,1,4]
    """
    sorted_ptr = tf.cast(
        tf.reduce_sum(
            tf.maximum(
                0, tf.linalg.band_part(unsorted_relative, 0, -1)
            ), axis=-2
        ), tf.int32
    )
    
    """
    partial_pos[b,i,j,m] == unsorted_relative[b,i,j] if i<=m and j<=m, otherwise == 0 
    """
    partial_pos = tf.repeat(
        unsorted_relative[:, tf.newaxis, :, :], n, axis=-3)
    partial_pos = tf.linalg.band_part(
        tf.transpose(partial_pos, [0, 3, 2, 1]), 0, -1)
    partial_pos = tf.linalg.band_part(
        tf.transpose(partial_pos, [0, 2, 1, 3]), 0, -1)
    """
    partial_pos[b,m,i] == at the mth decoding step, the sorted position of the ith decoded token,
    where the ith decoded token is the ith token in the generation order under the permutation; if i>m this equals zero
    
    e.g. in the above example, partial_pos[b,:,:] == [[0,0,0,0,0],[0,1,0,0,0],[0,1,2,0,0],[0,2,3,1,0],[0,2,3,1,4]]
    """    
    partial_pos = tf.cast(tf.reduce_sum(tf.maximum(
        0, tf.transpose(partial_pos, [0, 3, 1, 2])), axis=-2), tf.int32)

    """
    The variable sorted_ptr is in sorted (i.e. left-to-right) partial positions but the pointer
    network reuses state and does not sort as decoding progresses (i.e. it inserts with respect to the unsorted positions)
    so we need to convert into unsorted ptr positions.
    
    Here unsorted_ptr[b,i] == x means that the ith token in the generation order
    under the permutation should insert to the xth token's right,
    where the xth token refers to the xth token in the generation order under the permutation.
    
    Note that unsorted_ptr[b,0] is not meaningful because the 0th token in any generation
    order is always the <start> token and never inserted, so it does not have a reference token where it should insert to.
    
    In the above example, unsorted_ptr[b,:] == [0,0,1,0,2]
    """
    unsorted_ptr = tf.argmin(
        tf.abs(sorted_ptr[..., tf.newaxis] - 1 - partial_pos), 
        axis=-1, 
        output_type=tf.int32
    )
    
    """
    The <start> token is never inserted so we slice out the first channel;
    in addition there are only n - 1 valid insertion locations
    
    Also recall that partial_pos corresponds to the positions of already-decoded tokens
    """
    return tf.one_hot(unsorted_ptr[:, 1:], n - 1), partial_pos[:, :-1, :-1]


@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, None, None], dtype=tf.float32)])
def permutation_to_relative(permutation):
    """Converts a permutation matrix to an unsorted relative position
    matrix for training a language model

    Arguments:
    
    permutation: tf.Tensor, shape = [batch, len, len]
        a permutation matrix that defines the order in which
        words should be inserted by the language model
        
        the first dimension is the batch dimension, 
        the last two dimensions contain the 2d permutation matrix
        
        each entry must be 0/1
        
        the top left element of the permutation matrix must be 1
        since <start> token must always be generated first
        
        the bottom right element of the permutation matrix must be 1
        since it corresponds to the <end> or <pad> token
        
    Returns:

    relative: tf.Tensor, shape = [batch, len-1, len-1, 3]
        one hot vector of unsorted relative positions of words
        inserted by a language model at each decoding step (masks later applied during attention)"""

    # make sure the permutation is an int or the below computation
    # does not make sense
    permutation = tf.cast(permutation, tf.int32)

    # see "permutation_to_pointer" for explanations
    sorted_relative = -tf.math.cumsum(
        permutation, axis=-1, exclusive=True) + tf.math.cumsum(
            permutation, axis=-1, exclusive=True, reverse=True)

    # see "permutation_to_pointer" for explanations
    unsorted_relative = tf.matmul(
        permutation, sorted_relative, transpose_b=True)

    # get the one hot distribution of relative positions; contains
    # a one at location i when [left, center, right]_i
    return tf.one_hot(unsorted_relative[..., :-1, :-1] + 1, 3)

def pt_permutation_to_relative_l2r(s0, s1, n):
    """Converts a l2r permutation matrix to a relative position
    matrix; only used in the Permutation Transformer when
    "RelativePositionalAttentionLayer" is used.

    Arguments:

    s0: batch size
    s1: sentence length
    n: clip of position difference

    Returns:

    relative: tf.Tensor, [s0, s1, s1, 2n+1]
        relative positions of words"""

    sorted_relative = tf.range(s1)[tf.newaxis, tf.newaxis, :]
    sorted_relative = tf.tile(sorted_relative, [s0, s1, 1])
    shift = tf.range(s1)[tf.newaxis, :, tf.newaxis]
    sorted_relative = sorted_relative - shift
    sorted_relative = tf.clip_by_value(sorted_relative, -n, n)
    sorted_relative = tf.cast(sorted_relative, tf.int32)
    
    # get the one hot distribution of relative positions; contains
    # a one at location i when [left, center, right]_i
    return tf.one_hot(sorted_relative + n, 
                     tf.cast(2*n+1, tf.int32))


@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, None], dtype=tf.float32),
    tf.TensorSpec(shape=[None, None], dtype=tf.int32),
    tf.TensorSpec(shape=None, dtype=tf.string)])
def get_permutation(mask, words, order):
    """Construct a discrete permutation matrix for training a non monotonic
    autoregressive model using gradient descent

    Arguments:

    mask: tf.Tensor, shape = [batch, len]
        a tensor containing zeros and ones which indicate which elements
        of words are out of bounds
    words: tf.Tensor, shape = [batch, len]
        the batch of word ids that will be used to determine the
        permutation when using rare or common
    order: tf.Tensor
        the autoregressive ordering to train Transformer-InDIGO using;
        l2r, r2l, rare, or common

    Returns:

    permutation: tf.Tensor, shape = [batch, len, len]
        a permutation matrix for training a non monotonic autoregressive
        model using gradient descent"""

    # the dataset is not compiled with an ordering so one must
    # be generated on the fly during training; only applies
    # when using a pointer layer; note that the <end> token
    # must always be last and <start> token must always  be first
    b, n = tf.shape(words)[0], tf.shape(words)[1]

    if tf.equal(order, 'r2l'):  # corresponds to right-to-left
        length = tf.cast(tf.reduce_sum(mask, axis=1), tf.int32)
        ind = tf.tile(tf.range(n - 1)[tf.newaxis], [b, 1])
        ind = tf.reverse_sequence(ind, length - 2, seq_axis=1, batch_axis=0)
        ind = tf.concat([tf.fill([b, 1], 0), 1 + ind], axis=1)

    elif tf.equal(order, 'rare'):  # corresponds to rare-first
        upper_bound = tf.reduce_max(words, axis=1, keepdims=True) + 1
        scores = tf.where(tf.equal(words, 0), -tf.ones_like(words), words)
        scores = tf.where(tf.equal(words, 1), upper_bound, scores)
        scores = tf.where(tf.equal(words, 2), upper_bound + 1, scores)
        scores = tf.where(tf.equal(words, 3), tf.zeros_like(words), scores)
        ind = tf.argsort(scores, direction='DESCENDING')

    elif tf.equal(order, 'common'):  # corresponds to common-first
        upper_bound = tf.reduce_max(words, axis=1, keepdims=True) + 1
        scores = tf.where(tf.equal(words, 0), upper_bound + 2, words)
        scores = tf.where(tf.equal(words, 1), upper_bound, scores)
        scores = tf.where(tf.equal(words, 2), tf.zeros_like(words), scores)
        scores = tf.where(tf.equal(words, 3), upper_bound + 1, scores)
        ind = tf.argsort(scores, direction='ASCENDING')
    
    elif tf.equal(order, 'test'):
        ords = tf.concat([[0,6,1,4,7,3,2,5], tf.range(8, n)], axis=0)
        ind = tf.tile(ords[tf.newaxis], [b, 1])
        
    else:  # corresponds to left-to-right
        ind = tf.tile(tf.range(n)[tf.newaxis], [b, 1])

    return tf.one_hot(ind, n)

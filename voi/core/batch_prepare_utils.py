from voi.permutation_utils import permutation_to_pointer
from voi.permutation_utils import permutation_to_relative
from voi.permutation_utils import pt_permutation_to_relative_l2r
from voi.permutation_utils import get_permutation
from voi.common_enum import *
import tensorflow as tf
import numpy as np
import os
import copy

coco_batch_spec = [{
    'image_indicators': tf.TensorSpec(shape=[None, None], dtype=tf.float32),
    'image_path': tf.TensorSpec(shape=[None], dtype=tf.string),
    'tags': tf.TensorSpec(shape=[None, None], dtype=tf.int32),
    'words': tf.TensorSpec(shape=[None, None], dtype=tf.int32),
    'token_indicators': tf.TensorSpec(shape=[None, None], dtype=tf.float32),
    'global_features': tf.TensorSpec(shape=[None, None], dtype=tf.float32),
    'scores': tf.TensorSpec(shape=[None, None], dtype=tf.float32),
    'boxes': tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
    'labels': tf.TensorSpec(shape=[None, None], dtype=tf.int32),
    'boxes_features': tf.TensorSpec(shape=[None, None, None], dtype=tf.float32)}]

wmt_batch_spec = [{
    'encoder_words': tf.TensorSpec(shape=[None, None], dtype=tf.int32),
    'encoder_token_indicators': tf.TensorSpec(shape=[None, None], dtype=tf.float32),
    'decoder_words': tf.TensorSpec(shape=[None, None], dtype=tf.int32),
    'decoder_token_indicators': tf.TensorSpec(shape=[None, None], dtype=tf.float32)}]

wmt_with_tags_batch_spec = copy.deepcopy(wmt_batch_spec)
wmt_with_tags_batch_spec[0]['decoder_tags'] = tf.TensorSpec(shape=[None, None], dtype=tf.float32)
    
# @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.int32)]
#                              + coco_batch_spec)
def prepare_batch_for_lm_captioning(action_refinement, batch):
    """Transform a batch dictionary into a list of tensors
    for the autoregressive language model to process, for captioning datasets

    Arguments:
    
    action_refinement: tf.int32
        when training with VOI, the number of actions (permutations) to sample
        per training data
    batch: dict of tf.Tensors
        a dictionary that contains tensors from a tfrecord dataset;
        this function assumes region-features are used

    Returns:

    inputs: list of Tensors
        the input to be passed into the autoregressive decoder language model
        with attributes necessary for also computing the loss function"""

    def repeat_tensor_list(lst, n):
        for i in range(len(lst)):
            if isinstance(lst[i], tf.Tensor):
                lst[i] = tf.repeat(lst[i], n, axis=0)
        return lst
                       
    # select all relevant features from the batch dictionary
    image_ind = batch["image_indicators"]
    boxes_features = batch["boxes_features"]
    boxes = batch["boxes"]
    detections = batch["labels"]
    words = batch["words"]
    mask = batch["token_indicators"]
    batch_size = tf.shape(mask)[0]
    
    inputs = [None for _ in range(LEN_INPUTS)]
    inputs[QUERIES] = words[:, :-1]
    inputs[VALUES] = tf.zeros([batch_size])
    inputs[QUERIES_MASK] = tf.greater(mask[:, :-1], 0)
    inputs[VALUES_MASK] = tf.greater(image_ind, 0)
    inputs[IDS] = words[:, 1:]
    inputs[POINTER_PROBS] = tf.zeros([batch_size])
    inputs[LOG_PROBS] = tf.zeros([batch_size])
    inputs[OBJECT_DETECTIONS] = detections
    inputs[OBJECT_FEATURES] = boxes_features
    inputs[OBJECT_BOXES] = boxes
    return repeat_tensor_list(inputs, action_refinement)

# @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.int32)]
#                              + wmt_batch_spec)
# if using tags, wmt_batch_spec is replaced with wmt_with_tags_batch_spec
def prepare_batch_for_lm_wmt(action_refinement, batch):
    """Transform a batch dictionary into a list of tensors
    for the autoregressive language model to process, for non-captioning datasets

    Arguments:
    
    action_refinement: tf.int32
        in policy gradient, the number of actions (permutations) to sample
        per training data
    batch: dict of tf.Tensors
        a dictionary that contains tensors from a tfrecord dataset

    Returns:

    inputs: list of Tensors
        the input to be passed into the autoregressive decoder language model
        with attributes necessary for also computing the loss function"""

    def repeat_tensor_list(lst, n):
        for i in range(len(lst)):
            if isinstance(lst[i], tf.Tensor):
                lst[i] = tf.repeat(lst[i], n, axis=0)
        return lst
    
    # select all relevant features from the batch dictionary                       
    encoder_words = batch["encoder_words"]
    encoder_token_ind = batch["encoder_token_indicators"]
    words = batch["decoder_words"]
    mask = batch["decoder_token_indicators"]     
    batch_size = tf.shape(mask)[0]
    
    inputs = [None for _ in range(LEN_INPUTS)]
    inputs[QUERIES] = words[:, :-1]
    inputs[VALUES] = encoder_words
    inputs[QUERIES_MASK] = tf.greater(mask[:, :-1], 0)
    inputs[VALUES_MASK] = tf.greater(encoder_token_ind, 0)
    inputs[IDS] = words[:, 1:]
    inputs[POINTER_PROBS] = tf.zeros([batch_size])
    inputs[LOG_PROBS] = tf.zeros([batch_size])
    inputs[OBJECT_DETECTIONS] = tf.zeros([batch_size])
    inputs[OBJECT_FEATURES] = tf.zeros([batch_size])
    inputs[OBJECT_BOXES] = tf.zeros([batch_size, 1]) 
    return repeat_tensor_list(inputs, action_refinement)

# @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.bool),
#                              tf.TensorSpec(shape=None, dtype=tf.int32)] 
#                              + coco_batch_spec)
def prepare_batch_for_pt_captioning(pretrain_done, action_refinement, batch):
    """Transform a batch dictionary into a list of tensors
    for the permutation transformer to process (if training with Variational Order Inference), 
    for captioning datasets

    Arguments:

    pretrain_done: tf.bool
        whether decoder pretraining has finished
    action_refinement: tf.int32
        in policy gradient, the number of actions (permutations) to sample
        per training data     
    batch: dict of tf.Tensors
        a dictionary that contains tensors from a tfrecord dataset;
        this function assumes region-features are used

    Returns:

    inputs: list of Tensors
        the input to be passed into the Permutation Transformer to generate permutations
    """

    # select all relevant features from the batch dictionary
    image_ind = batch["image_indicators"]
    boxes_features = batch["boxes_features"]
    boxes = batch["boxes"]
    detections = batch["labels"]
    words = batch["words"]
    batch_size = tf.shape(words)[0]
    
    start_end_or_pad = tf.logical_or(tf.equal(
        words, 0), tf.logical_or(tf.equal(words, 2), tf.equal(words, 3)))

    # unused in pt's decoder unless pt_relative_embedding=True    
    l2r_relative = pt_permutation_to_relative_l2r(tf.shape(words)[0],
                                                  tf.shape(words)[1],
                                                  tf.constant(10)) 
    
    permu_inputs = [None for _ in range(LEN_INPUTS)]
    permu_inputs[QUERIES] = words
    permu_inputs[VALUES] = None
    permu_inputs[QUERIES_MASK] = tf.logical_not(start_end_or_pad)
    permu_inputs[VALUES_MASK] = tf.greater(image_ind, 0)
    permu_inputs[PRETRAIN_DONE] = pretrain_done
    permu_inputs[ACTION_REFINEMENT] = action_refinement
    permu_inputs[RELATIVE_POSITIONS] = l2r_relative
    permu_inputs[OBJECT_DETECTIONS] = detections
    permu_inputs[OBJECT_FEATURES] = boxes_features
    permu_inputs[OBJECT_BOXES] = boxes  
    return permu_inputs

# @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.bool),
#                              tf.TensorSpec(shape=None, dtype=tf.int32)] 
#                              + wmt_batch_spec)
# if using tags, wmt_batch_spec is replaced with wmt_with_tags_batch_spec
def prepare_batch_for_pt_wmt(pretrain_done, action_refinement, batch):
    """Transform a batch dictionary into a list of tensors
    for the permutation transformer to process (if training with Variational Order Inference), 
    for non-captioning datasets

    Arguments:

    pretrain_done: tf.bool
        whether decoder pretraining has finished
    action_refinement: tf.int32
        in policy gradient, the number of actions (permutations) to sample
        per training data       
    batch: dict of tf.Tensors
        a dictionary that contains tensors from a tfrecord dataset

    Returns:

    inputs: list of Tensors
        the input to be passed into the Permutation Transformer to generate permutations
    """

    # select all relevant features from the batch dictionary
    encoder_words = batch["encoder_words"]
    encoder_token_ind = batch["encoder_token_indicators"]
    words = batch["decoder_words"]
    mask = batch["decoder_token_indicators"]   
    if "decoder_tags" in batch.keys():
        tags = batch["decoder_tags"]
    else:
        tags = None
    batch_size = tf.shape(words)[0]
    
    start_end_or_pad = tf.logical_or(tf.equal(
        words, 0), tf.logical_or(tf.equal(words, 2), tf.equal(words, 3)))

    # unused unless pt_relative_embedding=True        
    l2r_relative = pt_permutation_to_relative_l2r(tf.shape(words)[0],
                                                  tf.shape(words)[1],
                                                  tf.constant(10))
    
    permu_inputs = [None for _ in range(LEN_INPUTS)]
    permu_inputs[QUERIES] = words
    permu_inputs[VALUES] = encoder_words
    permu_inputs[QUERIES_MASK] = tf.logical_not(start_end_or_pad)
    permu_inputs[VALUES_MASK] = tf.greater(encoder_token_ind, 0)
    permu_inputs[PRETRAIN_DONE] = pretrain_done
    permu_inputs[ACTION_REFINEMENT] = action_refinement
    permu_inputs[RELATIVE_POSITIONS] = l2r_relative
    permu_inputs[TAGS] = tags
    
    return permu_inputs

def prepare_permutation_without_pt(batch,
                                   dataset,
                                   tgt_vocab_size):
    """
    When not training with Variational Order Inference (i.e. training with fixed orderings),  
    transform a batch dictionary into a list of tensors
    for the autoregressive language model to process and include
    the permutated position information along with correctly permutated labels

    Arguments:

    batch: dict of tf.Tensors
        a dictionary that contains tensors from a tfrecord dataset;
        this function assumes region-features are used
    dataset: str
        type of dataset (captioning or wmt)        
    tgt_vocab_size: tf.Tensor
        the number of words in the target vocabulary of the model; used in order
        to calculate labels for the language model logits

    Returns:

    inputs: list of Tensors
        the input to be passed into a transformer model with attributes
        necessary for also computing the loss function"""

    # process the dataset batch dictionary into the standard
    # model input format
    if dataset == 'captioning':
        prepare_batch = prepare_batch_for_lm_captioning
    elif dataset in ['wmt', 'django', 'gigaword']:
        prepare_batch = prepare_batch_for_lm_wmt
    inputs = prepare_batch(tf.constant(1), batch)

    # the order is fixed
    if dataset in ['wmt', 'django', 'gigaword']:
        bt, bw = batch['decoder_token_indicators'], batch['decoder_words']
    else:
        bt, bw = batch['token_indicators'], batch['words']
    inputs[PERMUTATION] = get_permutation(bt, bw, tf.constant('l2r'))

    # Please see voi/permutation_utils.py for detailed explanations
    # convert the permutation to absolute and relative positions    
    inputs[ABSOLUTE_POSITIONS] = inputs[PERMUTATION][:, :-1, :-1]
    inputs[RELATIVE_POSITIONS] = permutation_to_relative(inputs[PERMUTATION])

    # convert the permutation to label distributions
    # also records the partial absolute position at each decoding time step
    hard_pointer_labels, inputs[PARTIAL_POS] = permutation_to_pointer(inputs[PERMUTATION])
    inputs[POINTER_LABELS] = hard_pointer_labels
    inputs[LOGITS_LABELS] = tf.matmul(
        inputs[PERMUTATION][:, 1:, 1:],
        tf.one_hot(inputs[IDS], tf.cast(tgt_vocab_size, tf.int32))
    )

    return inputs    

def prepare_permutation(batch,
                        tgt_vocab_size,
                        order,
                        dataset,
                        pretrain_done,
                        action_refinement,
                        decoder=None):
    """
    When training with Variational Order Inference (i.e. using Permutation Transformer to output orderings),  
    transform a batch dictionary into a list of tensors
    for the autoregressive language model to process and include
    the permutated position information along with correctly permutated labels
    
    Returns inputs for the autoregressive transformer; also return results
    from calling the Permutation Transformer

    Arguments:

    batch: dict of tf.Tensors
        a dictionary that contains tensors from a tfrecord dataset;
        this function assumes region-features are used
    tgt_vocab_size: tf.Tensor
        the number of words in the target vocabulary of the model; used in order
        to calculate labels for the language model logits
    order: str or tf.keras.Model
        the autoregressive ordering to train the autoregressive decoder;
        str = prespecified fixed permutation  
        tf.keras.Model = Permutation Transformer with VOI
    dataset: str
        type of dataset (captioning, django, gigaword, or wmt)
    pretrain_done: tf.bool
        for policy gradient, whether decoder pretraining with uniform permutations has finished
    action_refinement: tf.int32
        in policy gradient, the number of actions (permutations) to sample
        per training data  
    decoder: tf.keras.Model
        the autoregressive decoder model

    Returns:

    inputs: list of Tensors
        the input to be passed into autoregressive decoder with attributes
        necessary for also computing the loss function
    permu_inputs: list of Tensors
        the input to the Permutation Transformer 
        ALONG WITH the stored results from calling the Permutation Transformer,
        i.e. permu_inputs[ACTIVATIONS, KL, LOG_PERMU_PROBS]
    """

    # process the dataset batch dictionary into the standard
    # model input format
    
    if dataset == 'captioning':
        words = batch['words']
        mask = batch['token_indicators']
        prepare_batch_for_lm = prepare_batch_for_lm_captioning
        prepare_batch_for_pt = prepare_batch_for_pt_captioning
    elif dataset in ['wmt', 'django', 'gigaword']:
        words = batch['decoder_words']
        mask = batch['decoder_token_indicators']
        prepare_batch_for_lm = prepare_batch_for_lm_wmt
        prepare_batch_for_pt = prepare_batch_for_pt_wmt
        
    inputs = prepare_batch_for_lm(action_refinement, batch)
    permu_inputs = None
    # when the order is fixed, get the fixed permutation from utils
    if order in ['r2l', 'l2r', 'rare', 'common', 'test']:
        inputs[PERMUTATION] = get_permutation(mask, words, tf.constant(order))

    # in VOI,
    # pass the training example through the Permutation Transformer
    # to obtain the hard permutations (stored in inputs[PERMUTATION]) 
    # and other infos like log probability (stored in permu_inputs)
    if isinstance(order, tf.keras.Model):  # corresponds to soft orderings
        permu_inputs = prepare_batch_for_pt(pretrain_done,
            action_refinement, batch)
        inputs[PERMUTATION], activations, kl, log_nom, log_denom = \
            order(permu_inputs, training=True)
        permu_inputs[ACTIVATIONS] = activations
        permu_inputs[KL] = kl
        permu_inputs[LOG_PERMU_PROBS] = log_nom - log_denom

    # Searched Adaptive Order
    if order == 'sao' and decoder is not None:
        cap, logp, rel_pos = adaptive_search(
            inputs, decoder, dataset,
            beam_size=8, max_iterations=200, return_rel_pos=True)
        pos = tf.argmax(rel_pos, axis=-1, output_type=tf.int32) - 1
        pos = tf.reduce_sum(tf.nn.relu(pos), axis=2)
        pos = tf.one_hot(pos, tf.shape(pos)[2], dtype=tf.float32)
        ind = tf.random.uniform([tf.shape(pos)[0], 1], maxval=7, dtype=tf.int32)
        # todo: make sure this is not transposed
        inputs[PERMUTATION] = tf.squeeze(tf.gather(pos, ind, batch_dims=1), 1)

    inputs[PERMUTATION] = tf.stop_gradient(inputs[PERMUTATION])

    # Please see voi/permutation_utils.py for detailed explanations
    # convert the permutation to absolute and relative positions
    inputs[ABSOLUTE_POSITIONS] = inputs[PERMUTATION][:, :-1, :-1]
    inputs[RELATIVE_POSITIONS] = permutation_to_relative(inputs[PERMUTATION])

    # convert the permutation to label distributions
    # also records the partial absolute position at each decoding time step
    hard_pointer_labels, inputs[PARTIAL_POS] = permutation_to_pointer(inputs[PERMUTATION])
    inputs[POINTER_LABELS] = hard_pointer_labels
    inputs[LOGITS_LABELS] = tf.matmul(
        inputs[PERMUTATION][:, 1:, 1:],
        tf.one_hot(inputs[IDS], tf.cast(tgt_vocab_size, tf.int32))
    )

    return inputs, permu_inputs

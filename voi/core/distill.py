from voi.data.load_captioning import faster_rcnn_dataset
from voi.data.load_wmt import wmt_dataset
from voi.algorithms.beam_search import beam_search
from voi.permutation_utils import permutation_to_pointer
from voi.permutation_utils import permutation_to_relative
from voi.permutation_utils import get_permutation
from voi.core.batch_prepare_utils import prepare_batch_for_lm_wmt
from voi.core.batch_prepare_utils import prepare_permutation_without_pt
from voi.common_enum import *
import nltk
import tensorflow as tf
import os
import numpy as np
import copy
    
def distill_dataset(tfrecord_folder,
                    batch_size,
                    beam_size,
                    model,
                    model_ckpt,
                    vocab,
                    dataset_type,
                    strategy,
                    distillation_save_path):
    """Sequence-level dataset distillation using beam search
    
    Arguments:

    tfrecord_folder: str
        the path to a folder that contains tfrecord files
        ready to be loaded from the disk
    batch_size: int
    beam_size: int
        the maximum number of beams to use when decoding in a
        single batch
    model: tf.keras.Model
        the autoregressive decoder to distill from
    model_ckpt: str
        the path to an existing model checkpoint
    vocab: Vocabulary
        the model vocabulary which contains mappings
        from words to integers
    dataset: str
        type of dataset      
    strategy: tf.distribute.Strategy
        the strategy to use when distributing a model across many gpus
        typically a Mirrored Strategy        
    distillation_save_path: str
        save path for distillation output files"""
    
    prepare_batch_wmt = lambda b: prepare_batch_for_lm_wmt(action_refinement=1, batch=b)
    prepare_permutation = prepare_permutation_without_pt

    # create a distillation pipeline
    dataset = wmt_dataset(tfrecord_folder, batch_size, shuffle=False)
    prepare_batch = prepare_batch_wmt
        
    dataset = strategy.experimental_distribute_dataset(dataset)
    
    def dummy_loss_function(b):
        inputs = prepare_permutation(b, dataset_type, vocab.size())
        inputs_clone = [tf.identity(x) for x in inputs]
        _ = model(inputs_clone)
        loss, _ = model.loss(inputs, training=True)

    @tf.function(input_signature=[dataset.element_spec])
    def wrapped_dummy_loss_function(b):
        # distribute the model across many gpus using a strategy
        # do this by wrapping the loss function using data parallelism
        strategy.run(dummy_loss_function, args=(b,))
    
    def decode_function(b):
        # perform beam search using the current model and also
        # get the log probability of sequence
        if dataset_type in ['wmt', 'django']:
            maxit = 150        
        elif dataset_type in ['gigaword']:
            maxit = 40
        inputs = prepare_batch(b)
        cap, logp = beam_search(
            inputs, model, dataset_type, beam_size=beam_size, max_iterations=maxit)
        cap = tf.strings.reduce_join(
            vocab.ids_to_words(cap), axis=2, separator=' ')
        src = tf.strings.reduce_join(
            vocab.ids_to_words(inputs[VALUES]), axis=1, separator=' ')
        return src, cap, logp

    @tf.function(input_signature=[dataset.element_spec])               
    def wrapped_decode_function(b):
        # distribute the model across many gpus using a strategy
        # do this by wrapping the loss function
        return strategy.run(decode_function, args=(b,))

    # run the model for a single forward pass
    # and load en existing checkpoint into the trained model
    for batch in dataset:
        wrapped_dummy_loss_function(batch)
        break
        
    print("----------Done initializing the weights of the model-----------") 
    model.load_weights(model_ckpt)
    print("----------Done loading the weights of the model-----------")    
        
    # loop through the entire dataset once (one epoch)
    b_idx = 0
    
    f1 = open(os.path.join(distillation_save_path, "src_distillation.BPE.txt"), "w")
    f2 = open(os.path.join(distillation_save_path, "tgt_distillation.BPE.txt"), "w")
    
    # eliminate all elements in the array whose 
    # batch dimension is zero
    def eliminate_empty(arr):
        result = []
        for x in arr:
            if x.shape[0] != 0:
                result.append(x)
        return result
        
    def parse_output(s):
        return s.decode("utf-8").replace(
                    "<pad>", "").replace("<start>", "").replace(
                    "<end>", "").replace("  ", " ").strip()
        
    for batch in dataset:
        print("Batch index", b_idx)
        b_idx += 1

        # process the dataset batch dictionary into the standard
        # model input format; perform beam search
        src, cap, log_p = wrapped_decode_function(batch)
        if strategy.num_replicas_in_sync == 1:
            src = src.numpy()
            cap = cap.numpy()
        else:
            # when evaluating on multi gpus, the data might be distributed
            # in a way such that some gpus receive empty inputs, 
            # i.e. the batch dimension is zero
            src = tf.concat(eliminate_empty(src.values), axis=0).numpy()
            cap = tf.concat(eliminate_empty(cap.values), axis=0).numpy()
            log_p = tf.concat(eliminate_empty(log_p.values), axis=0)

        # format the model predictions into a string
        for i in range(cap.shape[0]):
            if dataset_type in ['wmt', 'django', 'gigaword']:
                model_sentence = parse_output(cap[i, 0])
                print("{}: [p = {}] {}".format(i, 
                                               np.exp(log_p[i, 0].numpy()),
                                               model_sentence))
                print(parse_output(src[i]), file=f1)
                print(model_sentence, file=f2)
                
        f1.flush()
        f2.flush()
   
    f1.close()
    f2.close()
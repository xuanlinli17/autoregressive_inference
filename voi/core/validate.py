from voi.data.load_captioning import faster_rcnn_dataset
from voi.data.load_wmt import wmt_dataset
from voi.algorithms.beam_search import beam_search
from voi.algorithms.nucleus_sampling import nucleus_sampling
from voi.permutation_utils import permutation_to_pointer
from voi.permutation_utils import permutation_to_relative
from voi.permutation_utils import get_permutation
from voi.core.batch_prepare_utils import prepare_batch_for_lm_captioning, prepare_batch_for_lm_wmt
from voi.core.batch_prepare_utils import prepare_permutation_without_pt
from voi.common_enum import *
import nltk
import tensorflow as tf
import os
import numpy as np
import copy 
    
def validate_dataset(tfrecord_folder,
                     caption_ref_folder,
                     batch_size,
                     beam_size,
                     model,
                     model_ckpt,
                     vocab,
                     dataset_type,
                     strategy,
                     validation_output_save_path):
    """
    Arguments:

    tfrecord_folder: str
        the path to a folder that contains tfrecord files
        ready to be loaded from the disk
    caption_ref_folder: str
        the path to a folder that contains ground truth sentence files
        ready to be loaded from the disk; unused if the dataset type is not captioning
    batch_size: int
    beam_size: int
        the maximum number of beams to use when decoding in a
        single batch
    model: tf.keras.Model
        the autoregressive decoder model to be validated
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
    validation_output_save_path: str
        ref_caps_list.txt and hyp_caps_list.txt save path, if dataset is not captioning"""

    prepare_batch_captioning = lambda b: prepare_batch_for_lm_captioning(action_refinement=1, batch=b)
    prepare_batch_wmt = lambda b: prepare_batch_for_lm_wmt(action_refinement=1, batch=b)    
    prepare_permutation = prepare_permutation_without_pt

    # create a validation pipeline
    if dataset_type == 'captioning':
        dataset = faster_rcnn_dataset(tfrecord_folder, batch_size, shuffle=False)
        prepare_batch = prepare_batch_captioning
    elif dataset_type in ['wmt', 'django', 'gigaword']:
        dataset = wmt_dataset(tfrecord_folder, batch_size, shuffle=False)
        prepare_batch = prepare_batch_wmt
        
    dataset = strategy.experimental_distribute_dataset(dataset)
    
    def dummy_loss_function(b):
        # process the dataset batch dictionary into the standard
        # model input format
        inputs = prepare_permutation(b, dataset_type, vocab.size())
        inputs_clone = [tf.identity(x) for x in inputs]
        _ = model(inputs_clone)
        loss, inputs = model.loss(inputs, training=True)

    @tf.function(input_signature=[dataset.element_spec])
    def wrapped_dummy_loss_function(b):
        # distribute the model across many gpus using a strategy
        # do this by wrapping the loss function using data parallelism
        strategy.run(dummy_loss_function, args=(b,))
    
    def decode_function(b):
        # perform beam search using the current model and also
        # get the log probability of sequence
        if dataset_type == 'captioning':
            maxit = 40
        elif dataset_type in ['wmt', 'django']:
            maxit = 150        
        elif dataset_type in ['gigaword']:
            maxit = 40
        inputs = prepare_batch(b)
        cap, logp = beam_search(
            inputs, model, dataset_type, beam_size=beam_size, max_iterations=maxit)
#         cap, logp = nucleus_sampling(
#             inputs, model, dataset_type,
#             num_samples=beam_size, nucleus_probability=0.95,
#             max_iterations=maxit)
        cap = tf.strings.reduce_join(
            vocab.ids_to_words(cap), axis=2, separator=' ')
        return cap, logp

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

    # for captioning tasks because there are multiple references per image
    ref_caps = {}
    hyp_caps = {}
    # for non-captioning tasks because there is only one reference per source,
    # so we can pair each reference with each hypothesis through the same index
    ref_caps_list = []
    hyp_caps_list = []    
        
    if dataset_type in ['wmt', 'django']:
        from vizseq.scorers.bleu import BLEUScorer
        from vizseq.scorers.meteor import METEORScorer
        from vizseq.scorers.ter import TERScorer 
        scorers = [BLEUScorer, METEORScorer, TERScorer]
        scorers = [s(corpus_level=True, sent_level=True, n_workers=2, verbose=False, extra_args=None)
                   for s in scorers]
    elif dataset_type in ['gigaword']:
        from vizseq.scorers.rouge import Rouge1Scorer, Rouge2Scorer, RougeLScorer
        scorers = [Rouge1Scorer, Rouge2Scorer, RougeLScorer]
        scorers = [s(corpus_level=True, sent_level=True, n_workers=2, verbose=False, extra_args=None)
                   for s in scorers]        
    elif dataset_type in ['captioning']:
        from vizseq.scorers.bleu import BLEUScorer
        scorers = [BLEUScorer] # calculate the BLEU score till the current evaluation step, for debugging
        scorers = [s(corpus_level=True, sent_level=True, n_workers=2, verbose=False, extra_args=None)
                   for s in scorers]        
        
    # loop through the entire dataset once (one epoch)
    b_idx = 0
    
    # eliminate all elements in the array whose 
    # batch dimension is zero
    def eliminate_empty(arr):
        result = []
        for x in arr:
            if x.shape[0] != 0:
                result.append(x)
        return result
        
    for batch in dataset:
        b_idx += 1
        
        # for every element of the batch select the path that
        # corresponds to ground truth words
        if dataset_type == 'captioning':
            if strategy.num_replicas_in_sync == 1:
                paths = [x.decode("utf-8") for x in batch["image_path"].numpy()]
            else:
                paths = [x.decode("utf-8") for x in tf.concat(batch["image_path"].values, axis=0).numpy()]
            paths = [os.path.join(caption_ref_folder,  os.path.basename(x)[:-7] + "txt")
                     for x in paths]

            # iterate through every ground truth training example and
            # select each row from the text file
            for file_path in paths:
                with tf.io.gfile.GFile(file_path, "r") as f:
                    ref_caps[file_path] = [
                        x for x in f.read().strip().lower().split("\n")
                        if len(x) > 0]
                    # Note that it's important to strip() and rejoin the reference captions 
                    # with a single space because they might have multiple
                    # spaces between two tokens; 
                    # Also note that when we were training the model, we never output
                    # spaces. Only when we calculate the metrics, the output tokens
                    # are joined through spaces.
                    ref_caps[file_path] = [' '.join(nltk.word_tokenize(x.strip())) 
                                           for x in ref_caps[file_path]]
                        

        # process the dataset batch dictionary into the standard
        # model input format; perform beam search
        cap, log_p = wrapped_decode_function(batch)
        if strategy.num_replicas_in_sync == 1:
            cap = cap.numpy()
        else:
            # when evaluating on multi gpus, the data might be distributed
            # in a way such that some gpus receive empty inputs, 
            # i.e. the batch dimension is zero
            cap = tf.concat(eliminate_empty(cap.values), axis=0).numpy()
            log_p = tf.concat(eliminate_empty(log_p.values), axis=0)

        if dataset_type in ['wmt', 'django', 'gigaword']:
            if strategy.num_replicas_in_sync == 1:
                bdw = batch['decoder_words']
            else:
                bdw = tf.concat(batch['decoder_words'].values, axis=0)
            wmt_truth_sentences = tf.strings.reduce_join(
                vocab.ids_to_words(bdw), axis=1, separator=' ').numpy()

        # format the model predictions into a string
        for i in range(cap.shape[0]):
            if dataset_type == 'captioning':
                hyp_caps[paths[i]] = cap[i, 0].decode("utf-8").replace(
                    "<pad>", "").replace("<start>", "").replace(
                    "<end>", "").replace("  ", " ").strip()
                print("{}: [p = {}] {}".format(paths[i], 
                                               np.exp(log_p[i, 0].numpy()),
                                               hyp_caps[paths[i]]))
            elif dataset_type in ['wmt', 'django', 'gigaword']:
                label_sentence = wmt_truth_sentences[i].decode("utf-8").replace(
                    "<pad>", "").replace("<start>", "").replace(
                    "<end>", "").replace("  ", " ").strip()
                model_sentence = cap[i, 0].decode("utf-8").replace(
                    "<pad>", "").replace("<start>", "").replace(
                    "<end>", "").replace("  ", " ").strip()
                print("{}: [p = {}] {}".format(i, 
                                               np.exp(log_p[i, 0].numpy()),
                                               model_sentence))
                ref_caps_list.append(label_sentence)
                hyp_caps_list.append(model_sentence)
         
        # print out intermediate metrics after each batch
        if dataset_type == 'captioning':
            tmp_ref_caps_list = []
            tmp_hyp_caps_list = []
            for key in ref_caps.keys():
                tmp_ref_caps_list.append(ref_caps[key])
                tmp_hyp_caps_list.append(hyp_caps[key])
            for scorer in scorers:
                scores = scorer.score(tmp_hyp_caps_list, [*zip(*tmp_ref_caps_list)])
                print(f'Vizseq Corpus-level {scorer} till now: {scores.corpus_score}')
        elif dataset_type in ['wmt', 'django', 'gigaword']:
            scores = scorers[0].score(hyp_caps_list, [ref_caps_list])
            if dataset_type != 'gigaword':
                print(f'Corpus-level BLEU till now (for wmt/iwslt/gigaword this is not real score; please post-process): {scores.corpus_score}')            
            else:
                print(f'Corpus-level ROUGE-1 till now (for wmt/iwslt/gigaword this is not real score; please post-process): {scores.corpus_score}')                

    # calculate the final metrics
    if dataset_type == 'captioning':
        # convert the dictionaries into lists for nlg eval input format
        for key in ref_caps.keys():
            ref_caps_list.append(ref_caps[key])
            hyp_caps_list.append(hyp_caps[key])
            
        for scorer in scorers:
            scores = scorer.score(hyp_caps_list, [*zip(*ref_caps_list)])
            print(f'Vizseq Corpus-level {scorer}: {scores.corpus_score}') 
            print("Above BLEU is smoothed; below bleu is not smoothed")

        from nlgeval import NLGEval
        nlgeval = NLGEval(no_skipthoughts=True, no_glove=True)

        # compute several natural language generation metrics
        # note that coco uses tokenized evaluation
        metrics = nlgeval.compute_metrics([*zip(*ref_caps_list)], hyp_caps_list)
        for key in metrics.keys():
            print("Eval/{}".format(key), metrics[key])
    elif dataset_type in ['wmt', 'django', 'gigaword']:
        with open(os.path.join(validation_output_save_path, "ref_caps_list.txt"), "w") as f:
            for t in ref_caps_list:
                print(t, file=f)
        with open(os.path.join(validation_output_save_path, "hyp_caps_list.txt"), "w") as f:
            for t in hyp_caps_list:
                print(t, file=f)                
        for scorer in scorers:
            scores = scorer.score(hyp_caps_list, [ref_caps_list])
            print(f'''Corpus-level {scorer} \
                  (for wmt/iwslt/gigaword this is not real score; \
                  please run post-processing scripts): {scores.corpus_score}''')
        
        if dataset_type == 'django':
            tot_sentences = len(ref_caps_list)
            correct = 0
            for idx in range(len(ref_caps_list)):
                if ref_caps_list[idx] == hyp_caps_list[idx]:
                    correct += 1
            print("accuracy {} {}".format(correct, tot_sentences), correct / tot_sentences)       

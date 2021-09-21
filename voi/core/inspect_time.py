from voi.data.load import faster_rcnn_dataset
from voi.data.load_wmt import wmt_dataset
from voi.algorithms.adaptive_search import adaptive_search
from voi.permutation_utils import permutation_to_pointer
from voi.permutation_utils import permutation_to_relative
from voi.permutation_utils import pt_permutation_to_relative_l2r
from voi.permutation_utils import get_permutation
from voi.core.batch_prepare_utils import prepare_batch_for_lm_captioning, prepare_batch_for_lm_wmt
from voi.core.batch_prepare_utils import prepare_batch_for_pt_captioning, prepare_batch_for_pt_wmt
from voi.core.batch_prepare_utils import prepare_permutation
from voi.algorithms.levenshtein import levenshtein
from voi.common_enum import *
from scipy import stats
import tensorflow as tf
import os
import numpy as np
import nltk
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time

np.set_printoptions(threshold=np.inf)

def inspect_time_dataset(tfrecord_folder,
                         caption_ref_folder,
                         batch_size,
                         beam_size,
                         model,
                         model_ckpt,
                         order,
                         vocab,
                         tags_vocab,
                         strategy,
                         visualization_save_path,
                         dataset_type):
    """
    Arguments:

    tfrecord_folder: str
        the path to a folder that contains tfrecord files
        ready to be loaded from the disk
    caption_ref_folder: str
        the path to a folder that contains ground truth caption files
        ready to be loaded from the disk; unused if dataset type is not captioning
    batch_size: int
    beam_size: int
        the maximum number of beams to use when decoding in a
        single batch
    model: tf.keras.Model
        the autoregressive decoder model to be validated
    model_ckpt: str
        the path to an existing model checkpoint
    order: str or tf.keras.Model
        str = fixed, predefined order; 
        tf.keras.Model = using permutation transformer to output ordering,
            which was used to train the autoregressive language model
    vocab: Vocabulary
        the model vocabulary which contains mappings
        from words to integers
    tags_vocab: Vocabulary or None
        parts of speech tags vocabulary; 
        if not None, then parts of speech information was used to train the permutation
        transformer        
    strategy: tf.distribute.Strategy
        the strategy to use when distributing a model across many gpus
        typically a Mirrored Strategy
    visualization_save_path: str
        visualization save path for sentences and orderings
    dataset_type: str
        the type of dataset"""

    def pretty(s):
        return s.replace('_', ' ').title()

    # create a validation pipeline
    if dataset_type == 'captioning':
        dataset = faster_rcnn_dataset(tfrecord_folder, batch_size, shuffle=False)
        prepare_batch_for_lm = prepare_batch_for_lm_captioning
        prepare_batch_for_pt = prepare_batch_for_pt_captioning
    elif dataset_type in ['wmt', 'django', 'gigaword']:
        dataset = wmt_dataset(tfrecord_folder, batch_size, shuffle=False, use_tags=tags_vocab is not None)
        prepare_batch_for_lm = prepare_batch_for_lm_wmt
        prepare_batch_for_pt = prepare_batch_for_pt_wmt
    dataset = strategy.experimental_distribute_dataset(dataset)

    def dummy_loss_function(b):
        # process the dataset batch dictionary into the standard
        # model input format
        inputs, permu_inputs = prepare_permutation(
            b, vocab.size(),
            order, dataset_type,
            pretrain_done=tf.constant(True), 
            action_refinement=tf.constant(1), 
            decoder=model
        )
        inputs_clone = [tf.identity(x) for x in inputs]
        _ = model(inputs_clone)
        loss, inputs = model.loss(inputs, training=True)

    @tf.function(input_signature=[dataset.element_spec])
    def wrapped_dummy_loss_function(b):
        # distribute the model across many gpus using a strategy
        # do this by wrapping the loss function using data parallelism
        strategy.run(dummy_loss_function, args=(b,))

    # run the model for a single forward pass
    # and load en existing checkpoint into the trained model
    for batch in dataset:
        wrapped_dummy_loss_function(batch)
        break

    print("----------Done defining weights of model-----------")

    if tf.io.gfile.exists(model_ckpt):
        model.load_weights(model_ckpt)
    if tf.io.gfile.exists(model_ckpt.replace(".", ".pt.")):
        order.load_weights(model_ckpt.replace(".", ".pt."))

    # for captioning
    ref_caps = {}
    hyp_caps = {}
    gen_order_caps = {}
    # for non-captioning
    ref_caps_list = []
    hyp_caps_list = []
    gen_order_list = []

    order_words_raw = np.ones(vocab.size(), dtype=np.float32) * (-1e-4)
    num_words_raw = np.ones(vocab.size(), dtype=np.float32) * 1e-4

    # create data frames for global sequence-level statistics
    time_stats_df = pd.DataFrame(columns=[
        'Model',
        'Type',
        'Sequence Length',
        'Time'])

    def decode_function(b):
        # calculate the ground truth sequence for this batch; and
        # perform beam search using the current model
        # show several model predicted sequences and their likelihoods
        inputs = prepare_batch_for_lm(tf.constant(1), b)
        start_time_sao = tf.timestamp()
        with tf.control_dependencies([start_time_sao]):
            cap, logp, rel_pos = adaptive_search(
                inputs, model, dataset_type, beam_size=beam_size, max_iterations=50,
                return_rel_pos=True)
        with tf.control_dependencies([cap, logp, rel_pos]):
            stop_time_sao = tf.timestamp()
        permu = None
        start_time_pt = 0
        stop_time_pt = 0
        if isinstance(order, tf.keras.Model):  # corresponds to soft orderings
            permu_inputs = prepare_batch_for_pt(tf.constant(True), tf.constant(1), b)
            start_time_pt = tf.timestamp()
            with tf.control_dependencies([start_time_pt]):
                permu, _, _, _, _ = order(permu_inputs)
            with tf.control_dependencies([permu]):
                stop_time_pt = tf.timestamp()
        return cap, logp, rel_pos, permu, \
               stop_time_sao - start_time_sao, \
               stop_time_pt - start_time_pt

    @tf.function(input_signature=[dataset.element_spec])
    def wrapped_decode_function(b):
        # distribute the model across many gpus using a strategy
        # do this by wrapping the loss function
        return strategy.run(decode_function, args=(b,))

    # loop through the entire dataset once (one epoch)
    if dataset_type in ['wmt', 'django', 'gigaword']:
        f = open(visualization_save_path, "w")

    for b_num, batch in enumerate(dataset):

        if dataset_type in ['wmt', 'django', 'gigaword']:
            bw = batch["decoder_words"]
        elif dataset_type == 'captioning':
            bw = batch["words"]
        if strategy.num_replicas_in_sync == 1:
            batch_wordids = bw
        else:
            batch_wordids = tf.concat(bw.values, axis=0)

        if dataset_type == 'captioning':
            if strategy.num_replicas_in_sync == 1:
                paths = [x.decode("utf-8") for x in batch["image_path"].numpy()]
            else:
                paths = [x.decode("utf-8") for x in tf.concat(batch["image_path"].values, axis=0).numpy()]
            paths = [os.path.join(caption_ref_folder, os.path.basename(x)[:-7] + "txt")
                     for x in paths]

            # iterate through every ground truth training example and
            # select each row from the text file
            for file_path in paths:
                with tf.io.gfile.GFile(file_path, "r") as f:
                    ref_caps[file_path] = [
                        x for x in f.read().strip().lower().split("\n")
                        if len(x) > 0]

        # process the dataset batch dictionary into the standard
        # model input format; perform beam search
        cap, log_p, rel_pos, permu, sao_time, pt_time = wrapped_decode_function(batch)
        if strategy.num_replicas_in_sync == 1:
            caparr, logparr, relposarr, permuarr = [cap], [log_p], [rel_pos], [permu]
            sao_timearr, pt_timeparr = [sao_time], [pt_time]
        else:
            caparr, logparr, relposarr, permuarr = \
                cap.values, log_p.values, rel_pos.values, permu.values
            sao_timearr, pt_timeparr = \
                sao_time.values, pt_time.values
        #             cap = tf.concat(cap.values, axis=0)
        #             log_p = tf.concat(log_p.values, axis=0)
        #             rel_pos = tf.concat(rel_pos.values, axis=0)
        for nzip, tmp in enumerate(zip(caparr, logparr, relposarr, permuarr,
                                       sao_timearr, pt_timeparr)):
            cap, log_p, rel_pos, permu, sao_time, pt_time = tmp
            # get the absolute position because the output of decoder
            # is a list of words whose order is determined by the
            # relative position matrix
            pos = tf.argmax(rel_pos, axis=-1, output_type=tf.int32) - 1
            pos = tf.reduce_sum(tf.nn.relu(pos[:, :, 1:, 1:]), axis=2)
            pos = tf.one_hot(pos, tf.shape(pos)[2], dtype=tf.int32)

            # calculate the generation order of captions
            gen_order_cap = tf.squeeze(tf.matmul(pos, cap[..., tf.newaxis]), axis=-1)

            cap_id = cap

            # generate a mask over valid words
            mask = tf.cast(tf.math.logical_not(
                tf.math.equal(cap_id, 0)), tf.float32)

            cap = tf.strings.reduce_join(
                vocab.ids_to_words(cap), axis=2, separator=' ').numpy()
            gen_order_cap = tf.strings.reduce_join(
                vocab.ids_to_words(gen_order_cap), axis=2, separator=' ').numpy()

            # format the model predictions into a string; the evaluation package
            # requires input to be strings; not there will be slight
            # formatting differences between ref and hyp
            for i in range(cap.shape[0]):
                real_i = nzip * cap.shape[0] + i
                if dataset_type == 'captioning' and paths[real_i] not in hyp_caps:
                    print(real_i)
                    hyp_caps[paths[real_i]] = cap[i, 0].decode("utf-8")
                    gen_order_caps[paths[real_i]] = gen_order_cap[i, 0].decode("utf-8")

                    if isinstance(order, tf.keras.Model):
                        print(b_num, "PT Permutation:\n", permu[i].numpy())
                        print(b_num, "Ground truth: {} | PT: {}".format(
                            tf.strings.reduce_join(
                                vocab.ids_to_words(batch_wordids[real_i]),
                                separator=' ').numpy(),
                            tf.strings.reduce_join(
                                vocab.ids_to_words(tf.squeeze(
                                    tf.matmul(tf.cast(permu[i], tf.int32),
                                              batch_wordids[real_i][:, tf.newaxis]))),
                                separator=' ').numpy()))

                    for j in range(log_p.shape[1]):

                        print("{}: [p = {}] {} | {}".format(
                            paths[i],
                            np.exp(log_p[i, j].numpy()),
                            cap[i, j].decode("utf-8"),
                            gen_order_cap[i, j].decode("utf-8")))

                        print("Decoder Permutation:\n", pos[i, j].numpy())

                        # the length of the sentence as an independent variable
                        seq_len = int(mask[i, j].numpy().sum()) - 1  # get rid of the end token

                        time_stats_df = time_stats_df.append({
                            "Model": model_ckpt,
                            "Type": "SAO",
                            'Sequence Length': seq_len,
                            'Time': float(sao_time.numpy().sum())},
                            ignore_index=True)
                        time_stats_df = time_stats_df.append({
                            "Model": model_ckpt,
                            "Type": "VOI",
                            'Sequence Length': seq_len,
                            'Time': float(pt_time.numpy().sum())},
                            ignore_index=True)

                elif dataset_type != 'captioning':
                    if "<unk>" not in tf.strings.reduce_join(
                            vocab.ids_to_words(batch_wordids[real_i]),
                            separator=' ').numpy().decode("utf-8"):
                        hyp_caps_list.append(cap[i, 0].decode("utf-8"))
                        gen_order_list.append(gen_order_cap[i, 0].decode("utf-8"))

                        if isinstance(order, tf.keras.Model):
                            print("PT Permutation:\n", permu[i].numpy(), file=f)
                            print("Ground truth: {} | PT: {}".format(
                                tf.strings.reduce_join(
                                    vocab.ids_to_words(batch_wordids[real_i]),
                                    separator=' ').numpy(),
                                tf.strings.reduce_join(
                                    vocab.ids_to_words(tf.squeeze(
                                        tf.matmul(tf.cast(permu[i], tf.int32),
                                                  batch_wordids[real_i][:, tf.newaxis]))),
                                    separator=' ').numpy()), file=f)

                        for j in range(log_p.shape[1]):
                            print("[p = {}] {} | {}".format(np.exp(log_p[i, j].numpy()),
                                                            cap[i, j].decode("utf-8"),
                                                            gen_order_cap[i, j].decode("utf-8")), file=f)
                            print("Decoder Permutation:\n", pos[i, j].numpy(), file=f)

        # process the logged metrics about order
        time_stats_df.to_csv(f'{model_ckpt}_time_stats_df.csv')

        plt.clf()
        g = sns.relplot(x='Sequence Length',
                        y='Time',
                        hue='Type',
                        data=time_stats_df,
                        kind="line",
                        height=5,
                        aspect=2,
                        facet_kws={"legend_out": True})
        g.set(title='Search Times For SAO vs VOI')
        plt.savefig(f'{model_ckpt}_timing.png',
                    bbox_inches='tight')
        plt.close()

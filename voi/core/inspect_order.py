from voi.data.load_captioning import faster_rcnn_dataset
from voi.data.load_wmt import wmt_dataset
from voi.algorithms.beam_search import beam_search
from voi.algorithms.nucleus_sampling import nucleus_sampling
from voi.permutation_utils import permutation_to_pointer
from voi.permutation_utils import permutation_to_relative
from voi.permutation_utils import pt_permutation_to_relative_l2r
from voi.permutation_utils import get_permutation
from voi.core.batch_prepare_utils import prepare_batch_for_lm_captioning, prepare_batch_for_lm_wmt
from voi.core.batch_prepare_utils import prepare_batch_for_pt_captioning, prepare_batch_for_pt_wmt
from voi.core.batch_prepare_utils import prepare_permutation
from voi.algorithms.levenshtein import levenshtein
from voi.data.tagger import load_parts_of_speech
from voi.data.tagger import load_tagger
from voi.common_enum import *
from scipy import stats
import tensorflow as tf
import os
import numpy as np
import nltk
import copy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)

def inspect_order_dataset(tfrecord_folder,
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
                          dataset_type,
                          tagger_file):
    """
    Arguments:

    tfrecord_folder: str
        the path to a folder that contains tfrecord files
        ready to be loaded from the disk
    caption_ref_folder: str
        the path to a folder that contains ground truth sentence files
        ready to be loaded from the disk, for captioning datasets only
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
        the type of dataset
    tagger_file: str
        path to parts of speech tagger file for visualizing generation orders in captioning datasets
    """

    def pretty(s):
        return s.replace('_', ' ').title()

    if dataset_type == 'captioning':
        captioning_tagger = load_tagger(tagger_file)
        captioning_tagger_vocab = load_parts_of_speech()

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

    # for captioning dataset
    ref_caps = {}
    hyp_caps = {}
    gen_order_caps = {}
    # for non-captioning dataset
    ref_caps_list = []
    hyp_caps_list = []
    gen_order_list = []

    order_words_raw = np.ones(vocab.size(), dtype=np.float32) * (-1e-4)
    num_words_raw = np.ones(vocab.size(), dtype=np.float32) * 1e-4

    # create data frames for global sequence-level statistics
    global_stats_df = pd.DataFrame(columns=[
        'Model',
        'Type',
        'Sequence Length',
        'Levenshtein Distance',
        'Normalized Levenshtein Distance',
        'Spearman Rank Correlation'])

    # create data frames for local token-level statistics
    local_stats_df = pd.DataFrame(columns=[
        'Model',
        'Type',
        'Word',
        'Part Of Speech',
        'Distance',
        'Normalized Distance'])

    def decode_function(b):
        # perform beam search using the current model and
        # get the log probability of sequence
        # if the order is soft (i.e. nonsequential), also return
        # the ordering the VOI encoder predicts
        if dataset_type == 'captioning':
            maxit = 40
        elif dataset_type in ['wmt', 'django']:
            maxit = 150        
        elif dataset_type in ['gigaword']:
            maxit = 40        
        inputs = prepare_batch_for_lm(tf.constant(1), b)

        # demonstration of nucleus sampler
#         cap, logp, rel_pos = nucleus_sampling(
#             inputs, model, dataset_type,
#             num_samples=beam_size, nucleus_probability=0.5,
#             max_iterations=maxit, return_rel_pos=True)

        # older beam search
        cap, logp, rel_pos = beam_search(
           inputs, model, dataset_type, beam_size=beam_size, max_iterations=maxit,
           return_rel_pos=True)

        permu = None
        if isinstance(order, tf.keras.Model):  # corresponds to soft orderings
            permu_inputs = prepare_batch_for_pt(tf.constant(True), tf.constant(1), b)
            permu, _, _, _, _ = order(permu_inputs)
        return cap, logp, rel_pos, permu

    @tf.function(input_signature=[dataset.element_spec])
    def wrapped_decode_function(b):
        # distribute the model across many gpus using a strategy
        # do this by wrapping the loss function
        return strategy.run(decode_function, args=(b,))
    
    if dataset_type in ['wmt', 'django', 'gigaword']:
        f = open(visualization_save_path, "w")
    elif dataset_type == 'captioning':
        reg, ext = visualization_save_path.split('.')
        f = open(reg + "_sentences" + "." + ext, "w")

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
                with tf.io.gfile.GFile(file_path, "r") as ftmp:
                    ref_caps[file_path] = [
                        x for x in ftmp.read().strip().lower().split("\n")
                        if len(x) > 0]

        # process the dataset batch dictionary into the standard
        # model input format; perform beam search
        cap, log_p, rel_pos, permu = wrapped_decode_function(batch)
        if strategy.num_replicas_in_sync == 1:
            caparr, logparr, relposarr, permuarr = [cap], [log_p], [rel_pos], [permu]
        else:
            caparr, logparr, relposarr, permuarr = (
                cap.values, log_p.values, rel_pos.values, permu.values
            )
        #             cap = tf.concat(cap.values, axis=0)
        #             log_p = tf.concat(log_p.values, axis=0)
        #             rel_pos = tf.concat(rel_pos.values, axis=0)
        sum_capshape0 = 0
        for nzip, tmp in enumerate(zip(caparr, logparr, relposarr, permuarr)):
            cap, log_p, rel_pos, permu = tmp
            sum_capshape0 += cap.shape[0]
            # get the absolute position because the output of decoder
            # is a list of words whose order is determined by the 
            # relative position matrix
            pos = tf.argmax(rel_pos, axis=-1, output_type=tf.int32) - 1
            pos = tf.reduce_sum(tf.nn.relu(pos[:, :, 1:, 1:]), axis=2)
            pos = tf.one_hot(pos, tf.shape(pos)[2], dtype=tf.int32)

            # calculate the generation order of captions
            gen_order_cap = tf.squeeze(tf.matmul(pos, cap[..., tf.newaxis]), axis=-1)

            # update stats of the generation order of each word
            word_ratio = tf.range(tf.shape(gen_order_cap)[-1], dtype=tf.float32)[tf.newaxis, tf.newaxis, :]
            word_ratio *= 1.0 / (tf.reduce_sum(tf.cast(gen_order_cap != 0, tf.float32), axis=-1, keepdims=True) - 1.0)
            goc, wr = tf.reshape(gen_order_cap, [-1]), tf.reshape(word_ratio, [-1])
            np.add.at(order_words_raw, goc, wr)
            np.add.at(num_words_raw, goc, 1.0)

            cap_id = cap

            # generate a mask over valid tokens
            mask = tf.cast(tf.math.logical_not(
                tf.math.equal(cap_id, 0)), tf.float32)

            cap = tf.strings.reduce_join(
                vocab.ids_to_words(cap), axis=2, separator=' ').numpy()
            gen_order_cap = tf.strings.reduce_join(
                vocab.ids_to_words(gen_order_cap), axis=2, separator=' ').numpy()

            # format the model predictions into a string; the evaluation package
            # requires input to be strings; note there will be slight
            # formatting differences between ref and hyp          
            for i in range(cap.shape[0]):
                real_i = sum_capshape0 - cap.shape[0] + i
                if dataset_type == 'captioning' and paths[real_i] not in hyp_caps:
                    print("Batch ID: ", real_i, log_p.shape)
                    hyp_caps[paths[real_i]] = cap[i, 0].decode("utf-8")
                    gen_order_caps[paths[real_i]] = gen_order_cap[i, 0].decode("utf-8")

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

                        print("{}: [p = {}] {} | {}".format(
                            paths[i],
                            np.exp(log_p[i, j].numpy()),
                            cap[i, j].decode("utf-8"),
                            gen_order_cap[i, j].decode("utf-8")), file=f)

                        print("Decoder Permutation:\n", pos[i, j].numpy(), file=f)

                        # evaluate the integer order induced by the decoder model
                        indices = np.argmax(pos[i, j].numpy(), axis=0)

                        # left to right permutation order
                        l2r = get_permutation(tf.concat([[1.0], mask[i, j]], 0)[tf.newaxis],
                                              tf.concat([[2], cap_id[i, j]], 0)[tf.newaxis],
                                              tf.constant("l2r"))[0, 1:, 1:]

                        # right to left permutation order
                        r2l = get_permutation(tf.concat([[1.0], mask[i, j]], 0)[tf.newaxis],
                                              tf.concat([[2], cap_id[i, j]], 0)[tf.newaxis],
                                              tf.constant("r2l"))[0, 1:, 1:]

                        # common first permutation order
                        cmn = get_permutation(tf.concat([[1.0], mask[i, j]], 0)[tf.newaxis],
                                              tf.concat([[2], cap_id[i, j]], 0)[tf.newaxis],
                                              tf.constant("common"))[0, 1:, 1:]

                        # rare first permutation order
                        rar = get_permutation(tf.concat([[1.0], mask[i, j]], 0)[tf.newaxis],
                                              tf.concat([[2], cap_id[i, j]], 0)[tf.newaxis],
                                              tf.constant("rare"))[0, 1:, 1:]

                        # the length of the sentence as an independent variable
                        seq_len = int(mask[i, j].numpy().sum()) - 1  # get rid of the end token
                        indices = indices[:seq_len]

                        # convert permutations into integer rank arrays
                        l2r = np.argmax(l2r.numpy(), axis=0)[:seq_len]
                        r2l = np.argmax(r2l.numpy(), axis=0)[:seq_len]
                        cmn = np.argmax(cmn.numpy(), axis=0)[:seq_len]
                        rar = np.argmax(rar.numpy(), axis=0)[:seq_len]

                        # compute rank correlation coefficients
                        l2r_c = stats.spearmanr(indices, l2r)[0]
                        r2l_c = stats.spearmanr(indices, r2l)[0]
                        cmn_c = stats.spearmanr(indices, cmn)[0]
                        rar_c = stats.spearmanr(indices, rar)[0]

                        # compute edit distances
                        l2r_d = levenshtein(indices, l2r)
                        r2l_d = levenshtein(indices, r2l)
                        cmn_d = levenshtein(indices, cmn)
                        rar_d = levenshtein(indices, rar)

                        # create data frames for analyzing the learned order
                        for a, l2r_i, r2l_i, cmn_i, rar_i, b in zip(
                                indices,
                                l2r,
                                r2l,
                                cmn,
                                rar,
                                cap_id[i, j].numpy()[:seq_len]):

                            # get the part of speech of this word
                            b = tf.convert_to_tensor([b])
                            b_word = vocab.ids_to_words(b)[0].numpy().decode("utf-8")
                            c_word = captioning_tagger.tag([b_word])[0][1]

                            local_stats_df = local_stats_df.append({
                                "Model": model_ckpt,
                                "Type": "Generation Index",
                                'Word': b_word,
                                'Part Of Speech': c_word,
                                'Location': a,
                                'Normalized Location': a / (seq_len - 1)},
                                ignore_index=True)
                            local_stats_df = local_stats_df.append({
                                "Model": model_ckpt,
                                "Type": "Left-To-Right Index",
                                'Word': b_word,
                                'Part Of Speech': c_word,
                                'Location': l2r_i,
                                'Normalized Location': l2r_i / (seq_len - 1)},
                                ignore_index=True)
                            local_stats_df = local_stats_df.append({
                                "Model": model_ckpt,
                                "Type": "Left-To-Right Distance",
                                'Word': b_word,
                                'Part Of Speech': c_word,
                                'Location': (a - l2r_i),
                                'Normalized Location': (a - l2r_i) / (seq_len - 1)},
                                ignore_index=True)
                            local_stats_df = local_stats_df.append({
                                "Model": model_ckpt,
                                "Type": "Right-To-Left Index",
                                'Word': b_word,
                                'Part Of Speech': c_word,
                                'Location': r2l_i,
                                'Normalized Location': r2l_i / (seq_len - 1)},
                                ignore_index=True)
                            local_stats_df = local_stats_df.append({
                                "Model": model_ckpt,
                                "Type": "Right-To-Left Distance",
                                'Word': b_word,
                                'Part Of Speech': c_word,
                                'Location': (a - r2l_i),
                                'Normalized Location': (a - r2l_i) / (seq_len - 1)},
                                ignore_index=True)
                            local_stats_df = local_stats_df.append({
                                "Model": model_ckpt,
                                "Type": "Common-First Index",
                                'Word': b_word,
                                'Part Of Speech': c_word,
                                'Location': cmn_i,
                                'Normalized Location': cmn_i / (seq_len - 1)},
                                ignore_index=True)
                            local_stats_df = local_stats_df.append({
                                "Model": model_ckpt,
                                "Type": "Common-First Distance",
                                'Word': b_word,
                                'Part Of Speech': c_word,
                                'Location': (a - cmn_i),
                                'Normalized Location': (a - cmn_i) / (seq_len - 1)},
                                ignore_index=True)
                            local_stats_df = local_stats_df.append({
                                "Model": model_ckpt,
                                "Type": "Rare-First Index",
                                'Word': b_word,
                                'Part Of Speech': c_word,
                                'Location': rar_i,
                                'Normalized Location': rar_i / (seq_len - 1)},
                                ignore_index=True)
                            local_stats_df = local_stats_df.append({
                                "Model": model_ckpt,
                                "Type": "Rare-First Distance",
                                'Word': b_word,
                                'Part Of Speech': c_word,
                                'Location': (a - rar_i),
                                'Normalized Location': (a - rar_i) / (seq_len - 1)},
                                ignore_index=True)

                        # update data frames for correlation
                        global_stats_df = global_stats_df.append({
                            "Model": model_ckpt,
                            "Type": "Left-To-Right Order",
                            'Sequence Length': seq_len,
                            'Spearman Rank Correlation': l2r_c,
                            'Levenshtein Distance': l2r_d,
                            'Normalized Levenshtein Distance': l2r_d / seq_len},
                            ignore_index=True)
                        global_stats_df = global_stats_df.append({
                            'Sequence Length': seq_len,
                            "Model": model_ckpt,
                            "Type": "Right-To-Left Order",
                            'Spearman Rank Correlation': r2l_c,
                            'Levenshtein Distance': r2l_d,
                            'Normalized Levenshtein Distance': r2l_d / seq_len},
                            ignore_index=True)
                        global_stats_df = global_stats_df.append({
                            "Model": model_ckpt,
                            "Type": "Common-First Order",
                            'Sequence Length': seq_len,
                            'Spearman Rank Correlation': cmn_c,
                            'Levenshtein Distance': cmn_d,
                            'Normalized Levenshtein Distance': cmn_d / seq_len},
                            ignore_index=True)
                        global_stats_df = global_stats_df.append({
                            "Model": model_ckpt,
                            "Type": "Rare-First Order",
                            'Sequence Length': seq_len,
                            'Spearman Rank Correlation': rar_c,
                            'Levenshtein Distance': rar_d,
                            'Normalized Levenshtein Distance': rar_d / seq_len},
                            ignore_index=True)

                elif dataset_type != 'captioning':
#                     if "<unk>" not in tf.strings.reduce_join(
#                             vocab.ids_to_words(batch_wordids[real_i]),
#                             separator=' ').numpy().decode("utf-8"):
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

    local_stats_df.to_csv(f'{model_ckpt}_local_stats_df.csv')
    global_stats_df.to_csv(f'{model_ckpt}_global_stats_df.csv')

    plt.clf()
    g = sns.relplot(x='Sequence Length',
                    y='Spearman Rank Correlation',
                    hue='Type',
                    data=global_stats_df,
                    kind="line",
                    height=5,
                    aspect=2,
                    facet_kws={"legend_out": True})
    g.set(title='Decoder Predictions Versus X')
    plt.savefig(f'{model_ckpt}_rank_correlation.png',
                bbox_inches='tight')

    plt.clf()
    g = sns.relplot(x='Sequence Length',
                    y='Levenshtein Distance',
                    hue='Type',
                    data=global_stats_df,
                    kind="line",
                    height=5,
                    aspect=2,
                    facet_kws={"legend_out": True})
    g.set(title='Decoder Predictions Versus X')
    plt.savefig(f'{model_ckpt}_edit_distance.png',
                bbox_inches='tight')

    plt.clf()
    g = sns.relplot(x='Sequence Length',
                    y='Normalized Levenshtein Distance',
                    hue='Type',
                    data=global_stats_df,
                    kind="line",
                    height=5,
                    aspect=2,
                    facet_kws={"legend_out": True})
    g.set(title='Decoder Predictions Versus X')
    plt.savefig(f'{model_ckpt}_normalized_edit_distance.png',
                bbox_inches='tight')

    # ------------

    if dataset_type == 'captioning':
        order_words = order_words_raw[4:]
        num_words = num_words_raw[4:]
        ratio_words = order_words / num_words
        all_words = vocab.ids_to_words(tf.range(4, vocab.size(), dtype=tf.int32)).numpy()

        arg_sorted = np.argsort(ratio_words)
        ratio_words = ratio_words[arg_sorted]
        order_words = order_words[arg_sorted]
        num_words = num_words[arg_sorted]
        all_words = all_words[arg_sorted]
        tagged_words = [nltk.pos_tag(nltk.word_tokenize(w.decode('UTF-8'))) for w in all_words]

        order_pospeech = {}
        num_pospeech = {}
        ratio_pospeech = {}

        with open(visualization_save_path, "w") as f:
            for i in range(ratio_words.shape[0]):
                if ratio_words[i] < 0.0:
                    continue
                word, pospeech = tagged_words[i][0]
                if pospeech in order_pospeech.keys():
                    order_pospeech[pospeech] += ratio_words[i] * num_words[i]
                    num_pospeech[pospeech] += num_words[i]
                else:
                    order_pospeech[pospeech] = ratio_words[i] * num_words[i]
                    num_pospeech[pospeech] = num_words[i]
                print(word, pospeech, int(num_words[i]), ratio_words[i], file=f)

            print("------------------------------", file=f)
            for pospeech in order_pospeech.keys():
                ratio_pospeech[pospeech] = order_pospeech[pospeech] / num_pospeech[pospeech]
            for k, v in sorted(ratio_pospeech.items(), key=lambda item: item[1]):
                print(k, int(num_pospeech[k]), v, file=f)

from voi.data.load import faster_rcnn_dataset
from voi.data.load_wmt import wmt_dataset
from voi.common_enum import *
from voi.algorithms.beam_search import beam_search
from voi.algorithms.adaptive_search import adaptive_search
from voi.permutation_utils import permutation_to_pointer
from voi.permutation_utils import permutation_to_relative
from voi.permutation_utils import pt_permutation_to_relative_l2r
from voi.permutation_utils import get_permutation
from voi.core.batch_prepare_utils import prepare_batch_for_lm_captioning, prepare_batch_for_lm_wmt
from voi.core.batch_prepare_utils import prepare_batch_for_pt_captioning, prepare_batch_for_pt_wmt
from voi.core.batch_prepare_utils import prepare_permutation
from voi.algorithms.levenshtein import levenshtein
from scipy import stats
import tensorflow as tf
import os
import numpy as np
import math
import nltk
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from matplotlib import transforms

np.set_printoptions(threshold=np.inf)

def perturb_features(tfrecord_folder,
                     caption_ref_folder,
                     batch_size,
                     beam_size,
                     model,
                     model_ckpt,
                     order,
                     vocab,
                     tags_vocab,
                     strategy,
                     dataset_type):
    """
    Arguments:

    tfrecord_folder: str
        the path to a folder that contains tfrecord files
        ready to be loaded from the disk
    caption_ref_folder: str
        the path to a folder that contains ground truth sentence files
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

    tf.io.gfile.makedirs(model_ckpt + '.perturb/')
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

    def visualize_function(b):
        # calculate the ground truth sequence for this batch; and
        # perform beam search using the current model
        # show several model predicted sequences and their likelihoods
        inputs = prepare_batch_for_lm(tf.constant(1), b)
        cap, logp, rel_pos = beam_search(
            inputs, model, dataset_type,
            beam_size=beam_size, max_iterations=50, return_rel_pos=True)

        pos = tf.argmax(rel_pos, axis=-1, output_type=tf.int32) - 1
        original_pos = tf.reduce_sum(tf.nn.relu(pos), axis=2)
        pos = tf.one_hot(original_pos, tf.shape(original_pos)[2], dtype=tf.float32)

        # select the most likely beam
        cap = cap[:, 0]
        pos = pos[:, 0]

        # the original cap is missing a start token
        inputs = prepare_batch_for_lm(tf.constant(1), b)
        full_cap = tf.concat([tf.fill([tf.shape(cap)[0], 1], START_ID), cap], 1)
        inputs[QUERIES] = full_cap[:, :-1]
        inputs[QUERIES_MASK] = tf.logical_not(tf.equal(inputs[QUERIES], 0))
        inputs[IDS] = full_cap[:,  1:]

        # todo: make sure this is not transposed
        inputs[PERMUTATION] = pos
        # convert the permutation to absolute and relative positions
        inputs[ABSOLUTE_POSITIONS] = inputs[PERMUTATION][:, :-1, :-1]
        inputs[RELATIVE_POSITIONS] = permutation_to_relative(inputs[PERMUTATION])

        # convert the permutation to label distributions
        # also records the partial absolute position at each decoding time step
        hard_pointer_labels, inputs[PARTIAL_POS] = permutation_to_pointer(inputs[PERMUTATION])
        inputs[POINTER_LABELS] = hard_pointer_labels
        inputs[LOGITS_LABELS] = tf.matmul(
            inputs[PERMUTATION][:, 1:, 1:],
            tf.one_hot(inputs[IDS], tf.cast(vocab.size(), tf.int32))
        )

        # perturb the image features by removing some of them
        original_mask = inputs[VALUES_MASK]
        range_i = tf.range(tf.shape(original_mask)[1])[tf.newaxis]
        original_pos = original_pos[:, 0]
        out_pos = original_pos[:, tf.newaxis]

        with tf.control_dependencies(inputs):

            for loc_i in tf.range(tf.shape(original_mask)[1]):

                # set the shape of out_pos
                tf.autograph.experimental.set_loop_options(
                    shape_invariants=[(
                        out_pos, tf.TensorShape([None, None, None]))])

                # create a mask that eliminates exactly one feature
                inputs[VALUES_MASK] = tf.logical_and(
                    original_mask, tf.not_equal(range_i, loc_i))

                # use searched adaptive order to
                # check how the predicted order changes
                perturb_rel_pos = adaptive_search(
                    inputs,
                    model,
                    dataset_type,
                    beam_size=beam_size,
                    max_iterations=50,
                    return_rel_pos=True)[2]

                # convert the rel_pos matrix to an array of positions
                perturb_pos = tf.argmax(
                    perturb_rel_pos, axis=-1, output_type=tf.int32) - 1
                perturb_pos = tf.reduce_sum(
                    tf.nn.relu(perturb_pos), axis=2)[:, 0]

                # if extra pad tokens are present at the end they will be removed
                # by adaptive search, and this places them back
                additional_pad_i = tf.range(tf.shape(original_pos)[1])[tf.newaxis]
                additional_pad_i = tf.broadcast_to(additional_pad_i, tf.shape(original_pos))
                additional_pad_i = additional_pad_i[:, tf.shape(perturb_pos)[1]:]
                perturb_pos = tf.concat([perturb_pos, additional_pad_i], 1)

                # add to the set of positions
                with tf.control_dependencies([out_pos, perturb_pos]):
                    out_pos = tf.concat([
                        out_pos, perturb_pos[:, tf.newaxis]], 1)

            # return the perturbed sequences and the Faster-RCNN boxes
            return full_cap, tf.cast(pos, tf.int32), out_pos, inputs[OBJECT_BOXES]

    @tf.function(input_signature=[dataset.element_spec])
    def wrapped_visualize_function(b):
        # distribute the model across many gpus using a strategy
        # do this by wrapping the loss function
        return strategy.run(visualize_function, args=(b,))

    example_id = 0
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
        cap, pos, gen_ord, boxes = wrapped_visualize_function(batch)
        if strategy.num_replicas_in_sync == 1:
            cap_arr, pos_arr, gen_ord_arr, boxes_arr = [cap], [pos], [gen_ord], [boxes]
        else:
            cap_arr, pos_arr, gen_ord_arr, boxes_arr = \
                cap.values, pos.values, gen_ord.values, boxes.values

        sum_capshape0 = 0
        for nzip, tmp in enumerate(zip(cap_arr,
                                       pos_arr,
                                       gen_ord_arr,
                                       boxes_arr)):
            cap, pos, gen_ord, boxes = tmp
            sum_capshape0 += cap.shape[0]

            cap_lens = tf.reduce_sum(tf.cast(tf.logical_or(
                tf.greater_equal(cap, 4),
                tf.equal(cap, 1)), tf.float32), axis=1)

            cap_seq = vocab.ids_to_words(cap)
            cap = tf.strings.reduce_join(
                vocab.ids_to_words(cap), axis=1, separator=' ')

            # format the model predictions into a string; the evaluation package
            # requires input to be strings; not there will be slight
            # formatting differences between ref and hyp
            for i in range(cap.shape[0]):
                real_i = sum_capshape0 - cap.shape[0] + i

                leni = int(cap_lens[i].numpy())
                gen_ordi = gen_ord[i, :, 1:(1 + leni)].numpy()
                weights = [levenshtein(gen_ordi[0], gen_ordi[j]) / leni
                           for j in range(1, gen_ordi.shape[0])]

                if dataset_type == 'captioning' and paths[real_i] not in hyp_caps:

                    hyp_caps[paths[real_i]] = cap[i].numpy().decode("utf-8")

                    img_path = paths[real_i].replace('captions_', '').replace('.txt', '.jpg')
                    img = plt.imread(img_path, format='jpg')

                    plt.clf()
                    fig = plt.figure(figsize=(10, 5))
                    gs = GridSpec(1, 2, figure=fig)
                    ax = fig.add_subplot(gs[:, :1])
                    ax.imshow(img, origin='upper')
                    ax.axis('off')
                    ax.set_title(f'Image ID: {os.path.basename(img_path)[:-4]}')

                    NX = 5

                    vec = np.array(weights)
                    vec_ids = np.argsort(vec)[::-1]
                    color_palette = ['#EE7733',
                                     '#0077BB',
                                     '#33BBEE',
                                     '#009988',
                                     '#CC3311',
                                     '#EE3377',
                                     '#BBBBBB',
                                     '#000000']

                    for which_idx, bi in enumerate(vec_ids[:NX]):

                        boxesi = boxes[i, bi].numpy()

                        ax.add_patch(patches.Rectangle(
                            (boxesi[0], 1.0 - boxesi[3]),
                            boxesi[2] - boxesi[0],
                            boxesi[3] - boxesi[1],
                            linewidth=2, edgecolor=color_palette[which_idx],
                            alpha=1.0 if vec[bi] > 0 else 0.0,
                            facecolor='none',
                            transform=ax.transAxes))

                    ax1 = fig.add_subplot(gs[:, 1:])
                    ax1.bar(np.arange(NX), vec[vec_ids[:NX]], color=color_palette[:NX])
                    ax1.spines['right'].set_visible(False)
                    ax1.spines['top'].set_visible(False)
                    ax1.yaxis.set_ticks_position('left')
                    ax1.xaxis.set_ticks_position('bottom')
                    ax1.set_title('Generation Order Sensitivity')
                    ax1.set_ylabel('Normalized Levenshtein Distance')
                    ax1.set_xlabel('Feature To Remove')
                    plt.savefig(model_ckpt + f'.perturb/{os.path.basename(img_path)[:-4]}.pdf')
                    plt.close()

                    print("Ground truth: {}".format(
                        tf.strings.reduce_join(
                            vocab.ids_to_words(batch_wordids[real_i]),
                            separator=' ').numpy().decode("utf-8")))
                    print("{}: {}".format(
                        paths[i], cap[i].numpy().decode("utf-8")))

                elif dataset_type != 'captioning':

                    if strategy.num_replicas_in_sync == 1:
                        encoder_words = batch["encoder_words"]
                    else:
                        encoder_words = tf.concat(
                            batch["encoder_words"].values, axis=0)

                    encoder_words = vocab.ids_to_words(encoder_words[real_i])

                    if "<unk>" not in tf.strings.reduce_join(
                            vocab.ids_to_words(batch_wordids[real_i]),
                            separator=' ').numpy().decode("utf-8"):

                        hyp_caps_list.append(cap[i].numpy().decode("utf-8"))

                        enci = [wi.decode('utf-8') for wi in encoder_words
                            .numpy().tolist() if wi not in [b'<start>', b'<end>', b'<pad>']]
                        capi = [wi.decode('utf-8') for wi in cap_seq[i]
                            .numpy().tolist() if wi not in [b'<start>', b'<end>', b'<pad>']]
                        colori = ['white' for wi in capi]

                        print("Ground truth: {}".format(
                            tf.strings.reduce_join(
                                vocab.ids_to_words(batch_wordids[real_i]),
                                separator=' ').numpy().decode("utf-8")))
                        print("{}".format(
                            cap[i].numpy().decode("utf-8")))

                example_id += 1

from voi.data.tagger import load_tagger
from voi.common_enum import *
from voi.process.captions import Sentence, Vocabulary
from collections import defaultdict
from dataclasses import dataclass
import pickle as pkl
import nltk
import os
import re
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

@dataclass
class LanguagePair(object):
    """A data class for storing information about
    the pair of sentences of two diff languages 
    in a single training example

    Arguments:

    source: Sentence
        source (e.g. de) contains the sentence pair
    target: Sentence
        target (e.g. en) contains the sentence pair
    """

    source: Sentence    
    target: Sentence  

def process_django(out_feature_folder,
                   in_feature_folder,
                   vocab_file,
                   max_length,
                   min_word_frequency,
                   dataset_type):
    """Generate the vocab files and process (sub)words into ids

    Arguments:

    out_feature_folder: str
        the path to a new folder where sentences represented using token ids
        will be pickled on the disk
    in_feature_folder: str
        a folder that contains source and target input text files
    vocab_file: str
        the path to a file that contains the model vocabulary
        and mappings from words to integers; 
        if the path doesn't exist then a new vocab file will be created
    max_length: int
        the maximum length of sentences from the dataset before
        cutting and ending the sentence early
    min_word_frequency: int
        the minimum frequency of words before adding such words to
        the model vocabulary
    dataset_type: str
        type of dataset"""

    # make the output folder if it does not exist already
    tf.io.gfile.makedirs(out_feature_folder)

    # get all the files
    src_file = open(os.path.join(in_feature_folder, "src_raw_" + dataset_type + ".txt"))
    tgt_file = open(os.path.join(in_feature_folder, "tgt_raw_" + dataset_type + ".txt"))

    # store frequencies and output words
    freq = defaultdict(int)
    src_freq = defaultdict(int)
    tgt_freq = defaultdict(int)
    all_src_words = []
    all_tgt_words = []

    # parse the entire set of captions
    while True:
        src = src_file.readline()
        tgt = tgt_file.readline()
        if not src:
            break
        src = src.strip()
        tgt = tgt.strip()
#         if "\\n" in tgt:
#             tgt = tgt.replace("\\n", " _newline_ ") # can't do this because of cases like "self . stream . write ( b'\n' )"
        src = re.sub(' +',' ',src)
        tgt = re.sub(' +',' ',tgt)
        src_list = src.split(' ')
        tgt_list = tgt.split(' ')
        if dataset_type != 'test' and (len(src_list) > 70 or len(tgt_list) > 50):
            continue
        for w in src_list:
            freq[w] += 1
            src_freq[w] += 1
        for w in tgt_list:
            freq[w] += 1
            tgt_freq[w] += 1

        all_src_words.extend([src_list])
        all_tgt_words.extend([tgt_list])

    def create_vocab_file(freq, min_word_frequency, vocab_file):
        if not tf.io.gfile.exists(vocab_file):
            # sort the dictionary using the frequencies as the key
            sorted_w, sorted_freq = list(
                zip(*sorted(freq.items(), key=(lambda x: x[1]), reverse=True)))

            # determine where to split the vocabulary
            split = 0
            for split, frequency in enumerate(sorted_freq):
                if frequency < min_word_frequency:
                    break

            # write the vocabulary file to the disk
            vocab = ("<pad>", "<unk>", "<start>", "<end>") + sorted_w[:(split + 1)]
            with tf.io.gfile.GFile(vocab_file, "w") as f:
                f.write("\n".join(vocab))
        else:
            # use an existing vocab file such as the training vocab file
            with tf.io.gfile.GFile(vocab_file, "r") as f:
                vocab = [x.strip() for x in f.readlines()]
        return Vocabulary(vocab, unknown_word="<unk>", unknown_id=UNK_ID)


    vocab = create_vocab_file(freq, min_word_frequency, 
                              os.path.join(os.path.dirname(vocab_file), os.path.basename(vocab_file)))

    # create mappings from words to integers
    data_list = []
    for index in range(len(all_src_words)):
        src_word_ids = np.concatenate(
                [[START_ID], vocab.words_to_ids(tf.constant(all_src_words[index])), [END_ID]], 0)
        tgt_word_ids = np.concatenate(
                [[START_ID], vocab.words_to_ids(tf.constant(all_tgt_words[index])), [END_ID]], 0)
        data_list.append(LanguagePair(Sentence(src_word_ids, None), Sentence(tgt_word_ids, None)))            

    sample_path = os.path.join(
        out_feature_folder, dataset_type + ".pkl")
    with tf.io.gfile.GFile(sample_path, "wb") as f:
        f.write(pkl.dumps(data_list))
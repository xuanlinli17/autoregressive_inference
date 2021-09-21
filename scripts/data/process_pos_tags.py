from voi.process.wmt import process_wmt
import argparse

from voi.process.captions import Sentence, Vocabulary
from voi.common_enum import *
from collections import defaultdict
from dataclasses import dataclass
import pickle as pkl
import nltk
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
        
def process_pos_tags(out_feature_folder,
                tag_file,
                vocab_file,
                dataset_type):
    """Process parts of speech tags into ids

    Arguments:

    out_feature_folder: str
        the path to a new folder where sentence features
        will be placed on the disk
    tag_file: str
        a file that contains the parts of speech tags to each token
        in each target sequence
    vocab_file: str
        the path to a file that contains the model vocabulary
        and mappings from words to integers
    dataset_type: str
        the type of dataset """

    print("""Please make sure to remove diacritics first in datasets like wmt16 ro-en""")
    # make the output folder if it does not exist already
    tf.io.gfile.makedirs(out_feature_folder)

    # get all the files
    tag_file = open(tag_file)

    # store frequencies and output words
    freq = defaultdict(int)
    all_tags = []

    # parse the entire set of captions
    tot_lines = 0
    while True:
        tot_lines += 1
        tag = tag_file.readline()
        if not tag:
            break
        tag = tag.strip()
        if tag[-1] == '\n':
            tag = tag[:-1]
        tag_list = tag.split(' ')

        for w in tag_list:
            freq[w] += 1

        all_tags.extend([tag_list])

    print("total_lines", tot_lines)
    
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
    
    vocab = create_vocab_file(freq, 1, 
        os.path.join(os.path.dirname(vocab_file), os.path.basename(vocab_file)))
    
    # create mappings from tags to integers
    data_list = []
    for index in range(len(all_tags)):
        tag_ids = np.concatenate(
                [[START_ID], vocab.words_to_ids(tf.constant(all_tags[index])), [END_ID]], 0)
        data_list.append(Sentence(tag_ids, None))

    sample_path = os.path.join(
        out_feature_folder, dataset_type + "_pos_tag.pkl")
    with tf.io.gfile.GFile(sample_path, "wb") as f:
        f.write(pkl.dumps(data_list))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out_feature_folder', type=str, default='/home/wmt/')
    parser.add_argument(
        "--tag_file", type=str, default='/home/wmt/tgt_distillation_tags.txt')
    parser.add_argument(
        '--vocab_file', type=str, default='/home/wmt/vocab_wmt_pos.txt')
    parser.add_argument(
        '--dataset_type', type=str, default='distillation', choices=['train', 'validation', 'test', 'distillation'])    
    args = parser.parse_args()

    process_pos_tags(args.out_feature_folder,
                     args.tag_file,
                     args.vocab_file,
                     args.dataset_type)

from voi.data.tagger import load_tagger
from voi.common_enum import *
from collections import defaultdict
from dataclasses import dataclass
import pickle as pkl
import nltk
import os
import numpy as np
import tensorflow as tf


@dataclass
class Sentence(object):
    """A data class for storing information about
    the words in a single training example

    Arguments:

    words: np.ndarray
        words ids that represent a single sentence with
        the starting and ending token
    tags: np.ndarray
        tag ids that represent a single sentence with
        the starting and ending token"""

    words: np.ndarray
    tags: np.ndarray


class Vocabulary(object):

    def __init__(self,
                 reverse_vocab,
                 unknown_word="<unk>",
                 unknown_id=1):
        """Creates a mapping from a discrete integer to the
        name of that element such as a part of speech

        Arguments:

        reverse_vocab: list
            a list of the words for this vocabulary object in
            order from highest frequency to lowest
        unknown_word: str
            the token already in reverse_vocab that corresponds to
            when a word is out of distribution
        unknown_id: int
            the id of the unknown token"""

        # create handles for numpy
        self.unknown_word = unknown_word
        self.unknown_id = unknown_id
        self.reverse_vocab = reverse_vocab
        self.vocab = {b: a for a, b in enumerate(reverse_vocab)}

        # create static tensors from the provided lists
        w = tf.constant(reverse_vocab)
        ids = tf.range(len(reverse_vocab))

        # create a lookup table for mapping between words and ids
        self.words_to_ids_hash = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=w,
                values=ids), default_value=tf.constant(unknown_id))

        # create a lookup table for mapping between words and ids
        self.ids_to_words_hash = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=ids,
                values=w), default_value=tf.constant(unknown_word))

    def size(self):
        """Return the size of the vocabulary

        Returns:

        size: int
            the number of words in the vocabulary"""

        return self.words_to_ids_hash.size()

    def words_to_ids(self,
                     keys):
        """Map a tensor of words into a tensor of ids

        Arguments:

        keys: tf.Tensor
            a tensor containing string word names

        Returns:

        ids: tf.Tensor
            the ids of words in keys"""

        return self.words_to_ids_hash.lookup(keys)

    def ids_to_words(self,
                     keys):
        """Map a tensor of ids into a tensor of words

        Arguments:

        keys: tf.Tensor
            a tensor containing integer ids

        Returns:

        names: tf.Tensor
            the names of words in keys"""

        return self.ids_to_words_hash.lookup(keys)


def process_captions(out_feature_folder,
                     in_folder,
                     tagger_file,
                     vocab_file,
                     max_length,
                     min_word_frequency):
    """Process captions in a specified folder from a standard format
    into numpy features using NLTK

    Arguments:

    out_feature_folder: str
        the path to a new folder where sentence features
        will be placed on the disk
    in_folder: str
        a folder that contains text files with multiple captions
        in a standard format
    tagger_file: str
        the path to a pre trained tagger file
    vocab_file: str
        the path to a file that contains the model vocabulary
        and mappings from words to integers;
        if this doesn't exist then a new file will be created
    max_length: int
        the maximum length of sentences from the dataset before
        cutting and ending the sentence early
    min_word_frequency: int
        the minimum frequency of words before adding such words to
        the model vocabulary"""

    tagger = load_tagger(tagger_file)

    # make the output folder if it does not exist already
    tf.io.gfile.makedirs(out_feature_folder)

    # get all the files to captions
    all_f = tf.io.gfile.glob(os.path.join(in_folder, "*.txt"))

    # store frequencies and output words
    freq = defaultdict(int)
    all_words = []
    all_tag = []

    # parse the entire set of captions
    for inner_path in all_f:

        # extract a list of captions for every caption file
        with tf.io.gfile.GFile(inner_path) as f:
            all_words.append([
                nltk.word_tokenize(line.strip().lower())[:max_length]
                for line in f.readlines() if len(line.strip()) > 0])

        # tag every word and count occurrences
        all_tag.append([])
        for line in all_words[-1]:
            all_tag[-1].append(list(zip(*tagger.tag(line)))[1])

            # increment the frequency of each word
            for word in line:
                freq[word] += 1

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

    # create mappings from words to integers
    vocab = Vocabulary(vocab, unknown_word="<unk>", unknown_id=UNK_ID)
    from voi.data.tagger import load_parts_of_speech
    parts_of_speech = load_parts_of_speech()

    # extract the ground truth ordering from the data
    for path_i, many_words, many_tag in zip(all_f, all_words, all_tag):

        # extract samples for every sentence for every data point
        samples = []
        for words_i, tag_i in zip(many_words, many_tag):

            # convert string names to integers
            words_ids = np.concatenate(
                [[START_ID], vocab.words_to_ids(tf.constant(words_i)), [END_ID]], 0)
            tag_ids = np.concatenate(
                [[START_ID], parts_of_speech.words_to_ids(tf.constant(tag_i)), [END_ID]], 0)

            # add the sample to the dataset
            samples.append(Sentence(words_ids, tag_ids))

        # write training examples to the disk by serializing their binary
        # these will be loaded later when building training examples
        sample_path = os.path.join(
            out_feature_folder, os.path.basename(path_i) + ".pkl")
        with tf.io.gfile.GFile(sample_path, "wb") as f:
            f.write(pkl.dumps(samples))

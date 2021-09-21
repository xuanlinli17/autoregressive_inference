import nltk
import pickle as pkl
import tensorflow as tf
import os
from voi.common_enum import *


def create_tagger(out_tagger_file: str):
    """Creates an instance of an NLTK tagger used for tagging
    the parts of speech of words in a data set 
    (we currently only use this for the visualization of generation orders in the captioning datasets)

    Arguments:

    out_tagger_file: str
        the path to a new tagger file that will be cached for
        future use during data processing and training"""

    # create a directory to house the tagger
    import pdb; pdb.set_trace()
    tf.io.gfile.makedirs(os.path.dirname(out_tagger_file))
    tagged_sent = nltk.corpus.brown.tagged_sents(tagset='universal')

    # split the training data into training and validation
    split = int(len(tagged_sent) * 0.9)
    training_sent = tagged_sent[:split]
    validation_sent = tagged_sent[split:]

    # build several staged tagger models using NLTK
    t0 = nltk.DefaultTagger('<unk>')
    t1 = nltk.UnigramTagger(training_sent, backoff=t0)
    t2 = nltk.BigramTagger(training_sent, backoff=t1)
    t3 = nltk.TrigramTagger(training_sent, backoff=t2)

    # evaluate the accuracy of each model
    scores = [[t0.evaluate(validation_sent), t0],
              [t1.evaluate(validation_sent), t1],
              [t2.evaluate(validation_sent), t2],
              [t3.evaluate(validation_sent), t3]]

    # choose the best model
    best_score, best_tagger = max(scores, key=lambda x: x[0])
    with tf.io.gfile.GFile(out_tagger_file, 'wb') as f:
        pkl.dump(best_tagger, f)


parts_of_speech = [
    "<pad>", "<unk>", "<start>", "<end>", ".", "CONJ", "DET", "ADP",
    "PRT", "PRON", "ADV", "NUM", "ADJ", "VERB", "NOUN"]


def load_parts_of_speech():
    """Loads the parts of speech into a vocabulary object
    for use in a tensorflow graph

    Returns:

    vocab: Vocabulary
        a vocabulary object for mapping between words and
        ids in a tensorflow static graph"""

    from voi.process.captions import Vocabulary
    return Vocabulary(
        parts_of_speech, unknown_word="<unk>", unknown_id=UNK_ID)


def load_tagger(tagger_file):
    """Loads a pre trained tagger file from the disk

    Arguments:

    tagger_file: str
        the path to a pre trained tagger file

    Returns:

    tagger: nltk.Tagger
        a nltk tagger for tagging parts of speech"""

    with tf.io.gfile.GFile(tagger_file, 'rb') as f:
        return pkl.loads(f.read())

import tensorflow as tf
import os
import pickle as pkl
import sys

def create_sequence_example_gigaword(inner_sample):
    """Creates a sequence example

    Arguments:
    inner_sample: Sentence
        a single training example with source and target sequences

    Returns:

    sequence_example: tf.train.SequenceExample
        a serializeable data format for TensorFlow"""

    src_words_feature = tf.train.FeatureList(
        feature=[tf.train.Feature(
            int64_list=tf.train.Int64List(value=[t])) for t in inner_sample.source.words])

    tgt_words_feature = tf.train.FeatureList(
        feature=[tf.train.Feature(
            int64_list=tf.train.Int64List(value=[t])) for t in inner_sample.target.words])

    # create the dictionary of features to save
    sequence_dict = dict(src_words=src_words_feature, tgt_words=tgt_words_feature)

    # create a sequence example
    return tf.train.SequenceExample(
        feature_lists=tf.train.FeatureLists(
            feature_list=sequence_dict))

def create_tfrecord_gigaword(out_tfrecord_folder,
                    feature_folder,
                    dataset_type,
                    samples_per_shard):
    """Create a sharded dataset by serializing features into tf
    records for efficient training/validation/testing

    Arguments:

    out_tfrecord_folder: str
        the string path to the directory to write tf record files
    feature_folder: str
        the string path to the folder containing the processed output 
        from process_gigaword.py
    dataset_type: str
        whether it's train or validation set or test set
    samples_per_shard: int
        the number of examples to serialize in a single
        shard for disk efficiency"""

    # create the outpout directory if it does not already exist
    out_tfrecord_folder = os.path.join(out_tfrecord_folder, dataset_type)
    tf.io.gfile.makedirs(out_tfrecord_folder)

    # create the initial file writer
    shard = 0
    num_samples_so_far = 0
    writer = tf.io.TFRecordWriter(os.path.join(
        out_tfrecord_folder, "{:013d}.tfrecord".format(shard)))

    with open(os.path.join(feature_folder, dataset_type + ".pkl"), 'rb') as f:
        samples = pkl.load(f)

    # loop through every training example
    for sample in samples:

        # occasionally flush all out streams to the disk
        if num_samples_so_far >= samples_per_shard:
            sys.stdout.flush()
            writer.close()

            # make a new writer when samples_per_shard is reached
            shard += 1
            num_samples_so_far = 0
            writer = tf.io.TFRecordWriter(os.path.join(
                out_tfrecord_folder, "{:013d}.tfrecord".format(shard)))

        # serialize a single sequence example to the disk
        sequence_sample = create_sequence_example_gigaword(sample)
        writer.write(sequence_sample.SerializeToString())
        num_samples_so_far += 1

    # done processing so flush any remaining data
    sys.stdout.flush()
    writer.close()

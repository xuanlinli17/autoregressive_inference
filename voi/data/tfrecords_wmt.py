import tensorflow as tf
import os
import pickle as pkl
import sys

def create_sequence_example_wmt(inner_sample, inner_sample_tag):
    """Creates a sequence example

    Arguments:
    inner_sample: Sentence
        a single training example with source and target sequences
    inner_sample_tag: np.ndarray or None
        the parts of speech tagging of the target sequence; None if not used

    Returns:

    sequence_example: tf.train.SequenceExample
        a serializeable data format for TensorFlow"""

    
    src_words_feature = tf.train.FeatureList(
        feature=[tf.train.Feature(
            int64_list=tf.train.Int64List(value=[t])) for t in inner_sample.source.words])

    tgt_words_feature = tf.train.FeatureList(
        feature=[tf.train.Feature(
            int64_list=tf.train.Int64List(value=[t])) for t in inner_sample.target.words])
    
    if inner_sample_tag is not None:
        tgt_tags_feature = tf.train.FeatureList(
            feature=[tf.train.Feature(
                int64_list=tf.train.Int64List(value=[t])) for t in inner_sample_tag.words])    
        sequence_dict = dict(src_words=src_words_feature, 
                             tgt_words=tgt_words_feature,
                             tgt_tags=tgt_tags_feature)        
    else:
        tgt_tags_feature = None
        sequence_dict = dict(src_words=src_words_feature, 
                             tgt_words=tgt_words_feature)

    # create a sequence example
    return tf.train.SequenceExample(
        feature_lists=tf.train.FeatureLists(
            feature_list=sequence_dict))

def create_tfrecord_wmt(out_tfrecord_folder,
                    feature_folder,
                    dataset_type,
                    samples_per_shard,
                    tags_file):
    """Create a sharded dataset by serializing features into tf
    records for efficient training/validation/testing

    Arguments:

    out_tfrecord_folder: str
        the string path to the directory to write tf record files
    feature_folder: str
        the string path to the folder containing the processed output 
        from process_wmt.py
    dataset_type: str
        whether it's train or validation set or test set
    samples_per_shard: int
        the number of examples to serialize in a single
        shard for disk efficiency 
    tags_file: str
        path to pickle file that stores parts of speech for each word in 
        each target sentence; empty if not used"""

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
        
    if tags_file != '':
        with open(os.path.join(tags_file), 'rb') as f:
            sample_tags = pkl.load(f)    
    else:
        sample_tags = None

    # loop through every training example
    print("example src shape", samples[0].source.words.shape[0])
    print("example target shape", samples[0].target.words.shape[0])
    if sample_tags is not None:
        print("example tag shape", sample_tags[0].words.shape[0])
        print("example tag", sample_tags[0].words)
        
    for idx in range(len(samples)):
        sample = samples[idx]
        if sample_tags is not None:
            sample_tag = sample_tags[idx]
            assert sample.target.words.shape[0] == sample_tag.words.shape[0]
        else:
            sample_tag = None
            
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
        sequence_sample = create_sequence_example_wmt(sample, sample_tag)
        writer.write(sequence_sample.SerializeToString())
        num_samples_so_far += 1

    # done processing so flush any remaining data
    sys.stdout.flush()
    writer.close()

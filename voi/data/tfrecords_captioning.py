import tensorflow as tf
import os
import pickle as pkl
import sys


def create_sequence_example(inner_image_path,
                            inner_sample):
    """Creates a sequence example from a single image and caption pair and
    conserves disk space by not duplicating features

    Arguments:

    inner_image_path: str
        the string path to processed image features extracted
        using a faster rcnn model
    inner_sample: Sentence
        a single caption with parts of speech tags
        (the tags here are not used during training, but used for visualization
        during validation)

    Returns:

    sequence_example: tf.train.SequenceExample
        a serializeable data format for TensorFlow"""

    # serialize a pointer to the disk location of the image features
    # copying data for every training example would consume too much storage
    image_path_feature = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[bytes(inner_image_path, "utf-8")]))

    # add all other tokens to the tf record
    words_feature = tf.train.FeatureList(
        feature=[tf.train.Feature(
            int64_list=tf.train.Int64List(value=[t])) for t in inner_sample.words])
    tags_feature = tf.train.FeatureList(
        feature=[tf.train.Feature(
            int64_list=tf.train.Int64List(value=[t])) for t in inner_sample.tags])

    # create the dictionary of features to save
    context_dict = dict(image_path=image_path_feature)
    sequence_dict = dict(words=words_feature, tags=tags_feature)

    # create a sequence example
    return tf.train.SequenceExample(
        context=tf.train.Features(feature=context_dict),
        feature_lists=tf.train.FeatureLists(
            feature_list=sequence_dict))


def create_tfrecord(out_tfrecord_folder,
                    caption_folder,
                    image_folder,
                    samples_per_shard):
    """Create a sharded dataset by serializing features into tf
    records for efficient training/validation/testing

    Arguments:

    out_tfrecord_folder: str
        the string path to the directory to write tf record files
    caption_folder: str
        the string path to the folder containing caption features 
        that have already been processed
    image_folder: str
        the string path to the folder containing image features
        that have already been processed
    samples_per_shard: int
        the number of examples to serialize in a single
        shard for disk efficiency"""

    # create the outpout directory if it does not already exist
    tf.io.gfile.makedirs(out_tfrecord_folder)

    # obtain all caption feature files
    all_caption_f = sorted(
        tf.io.gfile.glob(os.path.join(caption_folder, "*.txt.pkl")))

    # obtain all image feature files
    all_image_f = sorted(
        tf.io.gfile.glob(os.path.join(image_folder, "*.jpg.pkl")))

    # create the initial file writer
    shard = 0
    num_samples_so_far = 0
    writer = tf.io.TFRecordWriter(os.path.join(
        out_tfrecord_folder, "{:013d}.tfrecord".format(shard)))

    # loop through every image which can have several captions
    # save features to the disk in tfrecord format
    # TODO: this should be made parallel
    for caption_f, image_f in zip(all_caption_f, all_image_f):
        with tf.io.gfile.GFile(caption_f, "rb") as f:
            samples = pkl.loads(f.read())

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
            sequence_sample = create_sequence_example(image_f, sample)
            writer.write(sequence_sample.SerializeToString())
            num_samples_so_far += 1

    # done processing so flush any remaining data
    sys.stdout.flush()
    writer.close()
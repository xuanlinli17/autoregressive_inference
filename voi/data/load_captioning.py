from voi.process.rcnn_features import RCNNFeatures
import tensorflow as tf
import os
import pickle as pkl


def load_features(path):
    """A function for loading serialized image features from the disk
    that were extracted from a faster rcnn model

    Arguments:

    path: str
        the path to extracted image features

    Returns:

    features: Features
        a named tuple containing faster rcnn features"""

    with tf.io.gfile.GFile(path.decode('utf-8'), "rb") as f:
        return pkl.load(f)


def parse_faster_rcnn_sequence_example(sequence_example):
    """Parse a single sequence example that was serialized
    from the disk and build a tensor

    Arguments:

    sequence_example: tf.train.SequenceExample
        a single training example on the disk

    Returns:

    out: dict
        a dictionary containing all the features in the
        sequence example"""

    # read the sequence example binary
    context, sequence = tf.io.parse_single_sequence_example(
        sequence_example,
        context_features={
            "image_path": tf.io.FixedLenFeature([], dtype=tf.string)},
        sequence_features={
            "words": tf.io.FixedLenSequenceFeature([], dtype=tf.int64),
            "tags": tf.io.FixedLenSequenceFeature([], dtype=tf.int64)})

    # load the faster rcnn image features from the disk
    global_features, boxes_features, boxes, labels, scores = tf.numpy_function(
        load_features,
        [context["image_path"]],
        [tf.float32, tf.float32, tf.float32, tf.int64, tf.float32])

    """
    create a padding mask
    after padding, token_indicators == 1 if token is not <pad>; 0 otherwise
    e.g. 
    <start> A dog . <end> <pad> <pad>
       1    1  1  1   1     0     0
    """    
    token_indicators = tf.ones(tf.shape(sequence["words"]), dtype=tf.float32)
    image_indicators = tf.ones(tf.shape(labels), dtype=tf.float32)

    # cast every tensor to the appropriate data type
    image_path = context["image_path"]
    global_features = tf.cast(global_features, tf.float32)
    boxes_features = tf.cast(boxes_features, tf.float32)
    boxes = tf.cast(boxes, tf.float32)
    labels = tf.cast(labels, tf.int32)
    scores = tf.cast(scores, tf.float32)
    words = tf.cast(sequence["words"], tf.int32)
    tags = tf.cast(sequence["tags"], tf.int32)

    # build a dictionary containing all features
    return dict(
        image_path=image_path,
        global_features=global_features,
        boxes_features=boxes_features,
        boxes=boxes,
        labels=labels,
        scores=scores,
        image_indicators=image_indicators,
        words=words,
        tags=tags,
        token_indicators=token_indicators)


def parse_faster_rcnn_tf_records(record_files):
    """Parse a list of tf record files into a dataset of tensors for
    training a caption model

    Arguments:

    record_files: list
        a list of tf record files on the disk

    Returns:

    dataset: tf.data.Dataset
        create a dataset for parallel training"""

    return tf.data.TFRecordDataset(record_files).map(
        parse_faster_rcnn_sequence_example,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)


def faster_rcnn_dataset(tfrecord_folder,
                        batch_size,
                        shuffle=True):
    """Builds an input data pipeline for training deep image
    captioning models using region features

    Arguments:

    tfrecord_folder: str
        the path to a folder that contains tfrecord files
        ready to be loaded from the disk
    batch_size: int
        the maximum number of training examples in a
        single batch
    shuffle: bool
        specifies whether to shuffle the training dataset or not;
        do not shuffle during validation

    Returns:

    dataset: tf.data.Dataset
        a dataset that can be iterated over"""

    # select all files from the disk that contain training examples
    record_files = tf.data.Dataset.list_files(
        os.path.join(tfrecord_folder, "*.tfrecord"))

    # in parallel read from the disk into training examples
    dataset = record_files.interleave(
        parse_faster_rcnn_tf_records,
        cycle_length=tf.data.experimental.AUTOTUNE,
        block_length=2,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,)

    # shuffle and pad the data into batches for training
    if shuffle:
        dataset = dataset.shuffle(batch_size * 100)
    dataset = dataset.padded_batch(batch_size, padded_shapes={
        "image_path": [],
        "global_features": [1280],
        "boxes_features": [None, 1024],
        "boxes": [None, 4],
        "labels": [None],
        "scores": [None],
        "image_indicators": [None],
        "words": [None],
        "tags": [None],
        "token_indicators": [None]})

    # this line makes data processing happen in parallel to training
    return dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)
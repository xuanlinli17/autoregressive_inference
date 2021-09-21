import tensorflow as tf
import os
import pickle as pkl

def wmt_dataset(tfrecord_folder,
                batch_size,
                shuffle=True,
                use_tags=False):
    """Builds an input data pipeline for training
       datasets where source and target inputs are both sequences,
       such as Django, Gigaword, and WMT.

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
    use_tags: bool
        whether to use the target-sequence parts of speech tagging information to train
        the Permutation Transformer

    Returns:

    dataset: tf.data.Dataset
        a dataset that can be iterated over"""

    # select all files from the disk that contain training examples
    record_files = tf.data.Dataset.list_files(
        os.path.join(tfrecord_folder, "*.tfrecord"))

    # in parallel read from the disk into training examples
    dataset = record_files.interleave(
        lambda record_files: parse_wmt_tf_records(record_files, use_tags),
        cycle_length=tf.data.experimental.AUTOTUNE,
        block_length=2,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,)

    # shuffle and pad the data into batches for training
    if shuffle:
        dataset = dataset.shuffle(batch_size * 100)
    padded_shapes={
        "encoder_words": [None],
        "encoder_token_indicators": [None],
        "decoder_words": [None],
        "decoder_token_indicators": [None]}     
    if use_tags:
        padded_shapes["decoder_tags"] = [None]
    dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes)        

    # this line makes data processing happen in parallel to training
    return dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)


def parse_wmt_tf_records(record_files, use_tags):
    """Parse a list of tf record files into a dataset of tensors for training

    Arguments:

    record_files: list
        a list of tf record files on the disk
    use_tags: bool
        whether to use the target-sequence parts of speech tagging information 
        to train the Permutation Transformer    

    Returns:

    dataset: tf.data.Dataset
        create a dataset for parallel training"""

    return tf.data.TFRecordDataset(record_files).map(
        lambda seq_ex: parse_wmt_sequence_example(seq_ex, use_tags),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

def parse_wmt_sequence_example(sequence_example, use_tags):
    """Parse a single sequence example that was serialized
    from the disk and build a tensor

    Arguments:

    sequence_example: tf.train.SequenceExample
        a single training example on the disk
    use_tags: bool
        whether to use the target-sequence parts of speech tagging information
        to train the Permutation Transformer      

    Returns:

    out: dict
        a dictionary containing all the features in the
        sequence example"""

    # read the sequence example binary
    seq_features = {"src_words": tf.io.FixedLenSequenceFeature([], dtype=tf.int64),
                    "tgt_words": tf.io.FixedLenSequenceFeature([], dtype=tf.int64)}    
    if use_tags:
        seq_features["tgt_tags"] = tf.io.FixedLenSequenceFeature([], dtype=tf.int64)
    context, sequence = tf.io.parse_single_sequence_example(
        sequence_example,
        sequence_features=seq_features)

    """
    create a padding mask
    after padding, token_indicators == 1 if token is not <pad>; 0 otherwise
    e.g. 
    <start> A dog . <end> <pad> <pad>
       1    1  1  1   1     0     0
    """    
    src_token_indicators = tf.ones(tf.shape(sequence["src_words"]), dtype=tf.float32)
    tgt_token_indicators = tf.ones(tf.shape(sequence["tgt_words"]), dtype=tf.float32)

    # cast every tensor to the appropriate data type
    src_words = tf.cast(sequence["src_words"], tf.int32)
    tgt_words = tf.cast(sequence["tgt_words"], tf.int32)
    if use_tags:
        tgt_tags = tf.cast(sequence["tgt_tags"], tf.int32)

    # build a dictionary containing all features
    return_dict = dict(
        encoder_words=src_words,
        encoder_token_indicators=src_token_indicators,
        decoder_words=tgt_words,
        decoder_token_indicators=tgt_token_indicators)
    if use_tags:
        return_dict["decoder_tags"] = tgt_tags
    return return_dict
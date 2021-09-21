from dataclasses import dataclass
from typing import Any
import tensorflow as tf


@dataclass
class AttentionInput(object):
    """Fields of a data class for computing multihead attention
    in voi.variables.attention.Attention

    Arguments:

    queries: tf.Tensor
        the Queries tensor in a multihead attention mechanism
        see 'Attention Is All You Need'
    keys: tf.Tensor
        the Keys tensor in a multihead attention mechanism
        see 'Attention Is All You Need'
    values: tf.Tensor
        the Values tensor in a multihead attention mechanism
        see 'Attention Is All You Need'

    queries_mask: tf.Tensor
        a boolean mask for the Queries tensor
        in a multihead attention mechanism
    values_mask: tf.Tensor
        a boolean mask for the Keys and Values tensor
        in a multihead attention mechanism

    _keras_mask: tf.Tensor
        a required placeholder for tf.layers.Sequential"""

    # these are required for the network
    queries: Any = None
    keys: Any = None
    values: Any = None

    # if left unassigned these will not mask anything
    queries_mask: Any = tf.constant([[True]])
    values_mask: Any = tf.constant([[True]])

    # this does not need to be set during construction
    _keras_mask: Any = None


@dataclass
class TransformerInput(object):
    """Fields of a data class for computing multihead attention
    in voi.transformer.Transformer

    Arguments:

    queries: tf.Tensor
        the Queries tensor in a multihead attention mechanism
        see 'Attention Is All You Need'
    values: tf.Tensor
        the Keys and Values tensor in a multihead attention mechanism
        see 'Attention Is All You Need'

    queries_mask: tf.Tensor
        a boolean mask for the Queries tensor
        in a multihead attention mechanism
    values_mask: tf.Tensor
        a boolean mask for the Keys and Values tensor
        in a multihead attention mechanism

    _keras_mask: tf.Tensor
        a required placeholder for tf.layers.Sequential"""

    # these are required for the network
    queries: Any = None
    values: Any = None

    # if left unassigned these will not mask anything
    queries_mask: Any = tf.constant([[True]])
    values_mask: Any = tf.constant([[True]])

    # this does not need to be set during construction
    _keras_mask: Any = None
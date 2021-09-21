import tensorflow as tf


class Layer(tf.keras.layers.Layer):

    def loss(self,
             inputs,
             **kwargs):
        """A function that implements a forward pass and calculates the
        loss function for this layers layer

        Arguments:

        inputs: list of tf.Tensor

        Returns:

        loss: tf.Tensor
            a tensor representing the contribution this layer makes to the
            total model loss function
        outputs: list of tensors
            manages inputs and outputs for layers variables;
            note that the layers directly update the "inputs" list"""
        
        # If loss is not implemented in the subclass, call the layer and return the inputs
        return tf.zeros([]), self.call(inputs, **kwargs)

    def greedy_search(self,
                      inputs,
                      closed,
                      **kwargs):
        """A function that implements a forward pass and updates the decoding
        partial sequence using greedy search

        Arguments:

        inputs: list of tf.Tensor
        closed: tf.Tensor
            a boolean tensor where true values indicate that a beam has
            finished decoding and should not be modified"""

        return self.call(inputs, **kwargs), closed

    def nucleus_sampling(self,
                         inputs,
                         closed,
                         nucleus_probability,
                         **kwargs):
        """A function that implements a forward pass and updates the decoding
        partial sequence using the nucleas sampling strategy

        Arguments:

        inputs: list of tf.Tensor
        closed: tf.Tensor
            a boolean tensor where true values indicate that a beam has
            finished decoding and should not be modified
        nucleus_probability: float
            the probability threshold used to determine the size
            of the nucleus set of tokens to sample from"""

        return self.call(inputs, **kwargs), closed

    def beam_search(self,
                    inputs,
                    closed,
                    last_beam_size,
                    beam_size,
                    **kwargs):
        """A function that implements a forward pass and updates the decoding
        partial sequence using a beam search

        Arguments:

        inputs: list of tf.Tensor
        closed: tf.Tensor
            a boolean tensor where true values indicate that a beam has
            finished decoding and should not be modified
        last_beam_size: int
            the number of beams that were expanded by the last layer in an
            autoregressive model
        beam_size: int
            the number of beams to be expanded by this layer in an
            autoregressive model"""

        return self.call(inputs, **kwargs), closed, last_beam_size

    def adaptive_search(self,
                        inputs,
                        closed,
                        last_beam_size,
                        beam_size,
                        natural_order_tokens,
                        natural_order_pos,
                        **kwargs):
        """A function that implements a forward pass and updates the decoding
        partial sequence using a beam search

        Arguments:

        inputs: list of tf.Tensor
        closed: tf.Tensor
            a boolean tensor where true values indicate that a beam has
            finished decoding and should not be modified
        last_beam_size: int
            the number of beams that were expanded by the last layer in an
            autoregressive model
        beam_size: int
            the number of beams to be expanded by this layer in an
            autoregressive model
        natural_order_tokens: tf.Tensor
            a batch of sequences representing the generation index of tokens
            in natural order that are yet to be decoded.
        natural_order_pos: tf.Tensor
            a batch of sequences representing the word ids of tokens
            in natural order that are yet to be decoded."""

        return (self.call(inputs, **kwargs), closed, last_beam_size,
               natural_order_tokens, natural_order_pos)

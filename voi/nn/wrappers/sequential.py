import tensorflow as tf


class Sequential(tf.keras.Sequential):

    def loss(self,
             inputs,
             **kwargs):
        """A function that implements a forward pass and calculates the
        loss function for this layers layer

        Arguments:

        inputs: list of Tensors
            a list that contains ground truth information used for
            calculating the loss function

        Returns:

        loss: tf.Tensor
            a tensor representing the contribution this layer makes to the
            total model loss function
        outputs: list of tf.Tensor
            manages inputs and outputs;
            note that the layers directly update the "inputs" list"""

        total_loss = tf.zeros([])
        for layer in self.layers:
            loss, inputs = layer.loss(inputs, **kwargs)
            total_loss = total_loss + loss
        return total_loss, inputs
    
    def call(self,
             inputs,
             **kwargs):
        for layer in self.layers:
            inputs = layer.call(inputs, **kwargs)
        return inputs

    def visualize(self,
                  inputs,
                  **kwargs):
        all_visualizations = []
        for layer in self.layers:
            if hasattr(layer, 'visualize'):
                inputs, visualizations = layer.visualize(inputs, **kwargs)
                all_visualizations.extend(visualizations)
            else:
                inputs = layer.call(inputs, **kwargs)
        return inputs, all_visualizations
        
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

        for layer in self.layers:
            inputs, closed = layer.greedy_search(inputs, closed, **kwargs)
        return inputs, closed

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

        for layer in self.layers:
            inputs, closed = layer.nucleus_sampling(
                inputs, closed, nucleus_probability, **kwargs)
        return inputs, closed

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

        for layer in self.layers:
            inputs, closed, last_beam_size = layer.beam_search(
                inputs, closed, last_beam_size, beam_size,  **kwargs)
        return inputs, closed, last_beam_size

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

        for layer in self.layers:
            (inputs, closed, last_beam_size, 
            natural_order_tokens, natural_order_pos) = layer.adaptive_search(
                inputs, closed, last_beam_size, beam_size,
                natural_order_tokens, natural_order_pos, **kwargs
            )
        return (inputs, closed, last_beam_size, 
               natural_order_tokens, natural_order_pos)

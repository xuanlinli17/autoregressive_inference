from voi.nn.wrappers.layer import Layer
from voi.nn.base.block import Block
from voi.nn.base.sequence_to_mat_sinkhorn import SequenceToMatSinkhorn
from voi.nn.base.sinkhorn import Sinkhorn
from voi.nn.base.sinkhorn import matching
from voi.nn.base.bethe import Bethe
from voi.nn.input import AttentionInput
from voi.common_enum import *
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class PermutationSinkhornLayer(Layer):

    def __init__(self,
                 hidden_size,
                 heads,
                 queries_dropout=0.,
                 keys_dropout=0.,
                 iterations=200,
                 temperature=1.,
                 use_gumbel_noise=True,
                 hungarian_op_path='',
                 **kwargs):
        """Creates the last layer of Permutation Transformer 
        by applying a multihead sequence to the previous output matrix X. 
        Then sample permutations from the Gumbel Sinkhorn 
        distribution. Return the permutations and their probabilities.
        Arguments:
        hidden_size: int
            the number of units in the hidden variables used
            in each multi head attention layer
        heads: int
            the number of heads in each multi head attention layer
            a good default is 4 or 8
        queries_dropout: float
            the ratio of units to drop during training to the
            number of units in each attention layer
        keys_dropout: float
            the ratio of units to drop during training to the
            number of units in each attention layer
        iterations: tf.Tensor
            the total number of iterations of the Sinkhorn operator
            to apply to the data matrix
            (the number of Bethe iterations, which is used to 
            approximate the denominator of log-likelihood, is set to 30)
        temperature: float
            a positive number to divide the permutation logits,
            as in \tau in the original Gumbel Sinkhorn paper
        hungarian_op_path: string
            path to _hungarian_ops.so; if the path is invalid, then 
            tf.py_function(scipy.linear_sum_assignment) is used, which
            is not efficient in multi-gpu training.
        use_gumbel_noise: bool, UNUSED
            whether to apply gumbel noise to the output of PermutationLayer"""
        
        super(PermutationSinkhornLayer, self).__init__()
        
        try:
            self.hungarian_module = tf.load_op_library(hungarian_op_path)
            self.hungarian_fxn = self.hungarian_module.hungarian
        except Exception as e:
            print("Load hungarian module error:")
            print(e)
            print("Using scipy.optimize.linear_sum_assignment instead")
            # This is a lot slower due to tf.py_function not being able 
            # to parallelize across multiple gpus
            self.hungarian_module = matching
            self.hungarian_fxn = self.hungarian_module
        
        self.sinkhorn = Sinkhorn(iterations=iterations)
        self.bethe = Bethe(iterations=30)
        self.sequence_to_mat = SequenceToMatSinkhorn(
            queries_dropout=queries_dropout,
            keys_dropout=keys_dropout)
        self.block0 = Block(hidden_size // 2,
                            hidden_size * 2,
                            **kwargs)
        
        self.hidden_size = hidden_size
        self.heads = heads
        self.queries_dropout = queries_dropout
        self.keys_dropout = keys_dropout
        self.iterations = iterations
        self.temperature = temperature
        self.use_gumbel_noise = use_gumbel_noise
        self.kwargs = kwargs

    def call(self, inputs, **kwargs):
        """
        Arguments:
        inputs: list of Tensors
        Returns: A list of tensors below
            sample_permu:
                sampled hard permutations,
                with shape [batch * action_refinement, seq_length, seq_length]
            masked_refinement_acti: 
                the raw activation from the Permutation Transformer before passing through Sinkhorn, 
                i.e. X = q(y, x), where q is the Permutation Transformer
                after setting all entries in invalid positions 
                (i.e. (i,j) with either i or j corresponding to <start> <end> <pad>) 
                to be zero, no matter if i==j;
                the shape is [batch * action_refinement, seq_length, seq_length]
            kl: 
                KL((X+eps)/self.temperature || eps/temp_prior), 
                where X = q(y, x),
                      temp_prior = 1.0 and eps ~ gumbel noise;
                the shape is [batch * action_refinement, 1]
            log_nominator: 
                log of nominator of q(P|y, x), i.e. <X, P>_F
                where P = Hungarian(Sinkhorn((X + eps) / self.temperature))
                is the sampled permutation;
                the shape is [batch * action_refinement, 1]
            log_normalize_const: 
                log of denominator of q(P|y, x), approximated using Bethe permanent
                i.e. log perm(exp(X));
                the shape is [batch * action_refinement, 1]"""

        queries = inputs[QUERIES]
        queries_mask = inputs[QUERIES_MASK]   
        pretrain_done = inputs[PRETRAIN_DONE]
        action_refinement = inputs[ACTION_REFINEMENT]
        
        # calculate the shape of the values tensor before performing attention
        # used when separating the heads from channels
        shape = tf.shape(queries)
        hidden_dim = self.hidden_size // self.heads

        # apply final attention, equivalent to X <- X^THX
        activations = self.block0(queries, **kwargs)
        activations = tf.transpose(tf.reshape(activations, [
            shape[0], shape[1], self.heads, hidden_dim * 2]), [0, 2, 1, 3])

        queries_mask = tf.expand_dims(queries_mask, 1)
        attention_input = AttentionInput(
            queries=activations[..., :hidden_dim],
            keys=activations[..., hidden_dim:],
            queries_mask=queries_mask,
            values_mask=queries_mask)
        
        activations = self.sequence_to_mat(attention_input, **kwargs)
        activations = tf.reduce_sum(activations, axis=1)
        
        """
        Note that, for all positions i that are invalid, we set the output of the
        activations at (i,j) as -inf when i!=j and as zero if i=j. 
        This is because we need a valid permutation matrix that accounts
        for <start>, <end>, and <pad>, and <start> must be the the first token;
        <end> must be the last token before <pad>s, and <pad> must follow <end>.  
        
        e.g. for a sentence "<start> It's me . <end> <pad> <pad>",
        the activations become (-I = -inf; x = some non-(-inf) real)
        [[0, -I, -I, -I, -I, -I, -I],
         [-I, x, x, x, -I, -I, -I],
         [-I, x, x, x, -I, -I, -I],
         [-I, x, x, x, -I, -I, -I],
         [-I, -I,-I,-I, 0, -I, -I],
         [-I, -I,-I,-I, -I, 0, -I],
         [-I, -I,-I,-I, -I, -I, 0]]
        The doubly stochastic matrix after Gumbel-Sinkhorn will be
        [[1, 0, 0, 0, 0, 0, 0],
         [0, x, x, x, 0, 0, 0],
         [0, x, x, x, 0, 0, 0],
         [0, x, x, x, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 1]]      
        """
        valid_activation_mask = tf.cast(queries_mask, tf.float32)
        n_valid = tf.reduce_sum(valid_activation_mask, axis=-1)
        nsquare_valid = n_valid ** 2
        valid_activation_mask = tf.matmul(
            valid_activation_mask, valid_activation_mask,
            transpose_a = True
        )
        invalid_diags = tf.logical_and(
            tf.eye(shape[1], batch_shape=[shape[0]], dtype=tf.bool),
            tf.math.logical_not(tf.cast(valid_activation_mask, tf.bool))
        )
        invalid_diags = tf.cast(invalid_diags, tf.float32)
        activations = activations - activations * invalid_diags
        
        # if decoder pretrain not yet done, set activations in all valid positions
        # to be zero to allow uniform permutation sampling
        activations = tf.cond(
            pretrain_done, lambda: activations, 
            lambda: tf.add(activations, -activations * valid_activation_mask)
        )
        
        masked_activations = activations * valid_activation_mask

        # prevent the activation from being too large or too small
#         clipped_activations = tf.clip_by_value(masked_activations, -10000.0, 10000.0)
#         diff_activations = masked_activations - clipped_activations
#         activations = activations - diff_activations
#         masked_activations = activations * valid_activation_mask
        
        # calculate the kl divergence KL((X+eps)/self.temperature || eps/temp_prior), 
        # where eps ~ gumbel noise, temp_prior = 1.0
        kl_term1 = (
            nsquare_valid 
            * (
                tf.math.log(self.temperature) 
                - 1.0
                + np.euler_gamma * (
                    1.0 / self.temperature - 1.0)
              )
        )
        s1 = (
            1.0 
            / self.temperature 
            * tf.reshape(
                tf.reduce_sum(
                    masked_activations, axis=(-2,-1)), 
                (-1,1)
            )
        )
        # using tf.math.maximum for numerical stability
        s2 = tf.reshape(
            tf.reduce_sum(
                tf.math.exp(
                    -1.0 
                    / self.temperature 
                    * tf.math.maximum(
                        masked_activations, -20.0)
                ),
                axis=(-2,-1)
            ), 
            (-1,1)
        ) 
        s2 = s2 - (tf.cast(shape[1] * shape[1], tf.float32) - nsquare_valid)
        kl = (
            kl_term1 
            + s1 
            + s2 * tf.math.exp(
                tf.math.lgamma(1.0 + 1.0 / self.temperature)
            )
        )
        #tf.print("kl", tf.squeeze(kl), summarize=-1)
        kl = tf.repeat(kl, action_refinement, axis=0)

        # calculate the normalizing constant (log(denominator)) using Bethe iteration:
        bethe_activations = self.bethe(activations / self.temperature, **kwargs)
        masked_bethe_activations = bethe_activations * valid_activation_mask
        masked_bethe_activations = tf.stop_gradient(masked_bethe_activations)
        m1mba = tf.stop_gradient((1.0 - masked_bethe_activations) * valid_activation_mask)
        eps = 1e-10
        term1 = tf.reduce_sum(
            masked_bethe_activations 
            * masked_activations, 
            axis=(-2,-1)
        )
        term2 = -tf.reduce_sum(
            masked_bethe_activations 
            * tf.math.log(masked_bethe_activations + eps), 
            axis=(-2,-1)
        )
        term3 = tf.reduce_sum(
            m1mba 
            * tf.math.log(m1mba+eps), 
            axis=(-2,-1)
        )
        log_normalize_const = tf.reshape(term1 + term2 + term3, (-1, 1))

        # Repeat the normalizing constant by the number of permutations we sample
        # for each data
        log_normalize_const = tf.repeat(log_normalize_const, action_refinement, axis=0)
        
        # sample permutation using Gumbel noise and calculate log(nominator)
        action_refinement_acti = tf.repeat(activations, action_refinement, axis=0)
        g = tfp.distributions.Gumbel(
            loc=tf.zeros_like(action_refinement_acti), 
            scale=tf.ones_like(action_refinement_acti))
        sampled_noise = g.sample()        
        action_refinement_mask = tf.repeat(valid_activation_mask, action_refinement, axis=0)
        masked_refinement_acti = action_refinement_acti * action_refinement_mask
        sampled_noise = sampled_noise * action_refinement_mask
        sample_permu = self.sinkhorn(
            (action_refinement_acti + sampled_noise) 
            / self.temperature, 
            **kwargs
        )
        # If we use the hungarian op from https://github.com/mbaradad/munkres-tensorflow,
        # then this hungarian op requires all costs to be different or otherwise
        # bugs will occur, so we add a tiny noise to the doubly stochastic matrix.
        # sample_permu = sample_permu + tf.random.normal(tf.shape(sample_permu)) * 1e-7
        sample_permu = tf.one_hot(
            self.hungarian_fxn(-sample_permu), 
            depth=shape[1], 
            axis=-1
        )
        log_nominator = tf.linalg.trace(
            tf.matmul(
                masked_refinement_acti, 
                sample_permu,
                transpose_a=True
            )
        )
        log_nominator = tf.reshape(log_nominator, (-1, 1))

        return [sample_permu, masked_refinement_acti, kl, 
                log_nominator, log_normalize_const]

    
"""
Debugging and deleted stuff
"""
# tf.print("term1", term1[0], "term2", term2[0], "term3", term3[0])

# tf.print("pretrain_done", pretrain_done)
#tf.print("masked_activations", masked_activations[0], summarize=-1)
#tf.print("kl", tf.squeeze(kl), summarize=-1)
#tf.print("mean kl", tf.reduce_mean(kl), summarize=-1)
#       tf.print("bethe row column sum", tf.reduce_sum(bethe_activations, axis=(-1))[0],
#                                        tf.reduce_sum(bethe_activations, axis=(-2))[0],
#                                        summarize=-1)


# sample permutations to calculate log denominator
#         nsamples = 99
#         detach_acti = tf.stop_gradient(activations)
#         detach_acti = tf.repeat(detach_acti, nsamples, axis=0)
#         denom_g = tfp.distributions.Gumbel(
#             loc=tf.zeros_like(detach_acti), scale=tf.ones_like(detach_acti))
#         denom_noise = denom_g.sample()
#         denom_permu = self.sinkhorn((
#             detach_acti + denom_noise) / self.temperature, **kwargs) 
#         denom_permu = tf.cast(matching(denom_permu), tf.float32)
#         denom_activations = tf.repeat(masked_activations, nsamples, axis=0)
#         log_denominator = tf.linalg.trace(tf.matmul(denom_activations, denom_permu,
#                                           transpose_a=True))
#         log_denominator = tf.reshape(log_denominator, (-1, nsamples))
#         log_denominator = tf.concat([log_nominator, log_denominator], axis=1)
#         log_normalize_const = tf.math.reduce_logsumexp(log_denominator, axis=1, keepdims=True)
        
#         tf.print("log_nominator", tf.squeeze(log_nominator), 
#                  "log_denom", tf.squeeze(log_normalize_const), summarize=-1)
#         tf.print("term1", term1[0], "term2", term2[0], "term3", term3[0], 
#                  "log_nominator", log_nominator[0])


# zero-mean the activations
#         activation_avg = tf.reduce_sum(masked_activations, axis=(1,2))[:, tf.newaxis]
#         activation_avg = activation_avg / nsquare_valid
#         activations = activations - activation_avg[..., tf.newaxis] * valid_activation_mask
#         masked_activations = activations * valid_activation_mask


#         tf.print("---------------------------------------")
#         tf.print("masked activations[0]", masked_activations[0], summarize=-1) 
#         tf.print("masked activations[1]", masked_activations[1], summarize=-1)         
#         tf.print("masked sinkhorn activations[0]", masked_sinkhorn_activations[0], summarize=-1) 
#         tf.print("masked sinkhorn activations[1]", masked_sinkhorn_activations[1], summarize=-1)         
#         tf.print("acti * mask[0]", action_refinement_acti[0] * action_refinement_mask[0], summarize=-1)
#         tf.print("acti * mask[1]", action_refinement_acti[1] * action_refinement_mask[1], summarize=-1)        
#         tf.print("sample_permu[0]", sample_permu[0], summarize=-1)
#         tf.print("sample_permu[1]", sample_permu[1], summarize=-1)   
#         tf.print("---------------------------------------")


#tf.print("frob_norm", frob_norm[0], "ent_sinkhorn_acti", ent_sinkhorn_acti[0])
# tf.print("log_nominator", log_nominator[0])



#         # calculate the normalizing constant (log(denominator)) using Sinkhorn iteration:
#         sinkhorn_activations = self.sinkhorn(activations / self.temperature, **kwargs)
#         masked_sinkhorn_activations = sinkhorn_activations * valid_activation_mask
#         # prevent backprop thru sinkhorn operator
#         masked_sinkhorn_activations = tf.stop_gradient(masked_sinkhorn_activations)
#         tf.print("sinkhorn row column sum", tf.reduce_sum(sinkhorn_activations, axis=(-1))[0],
#                                             tf.reduce_sum(sinkhorn_activations, axis=(-2))[0],
#                                             summarize=-1)
#         frob_norm = tf.linalg.trace(tf.matmul(masked_activations, masked_sinkhorn_activations,
#                                               transpose_a=True))
#         eps = 1e-10
#         ent_sinkhorn_acti = -tf.reduce_sum((masked_sinkhorn_activations + eps) * tf.math.log(
#             masked_sinkhorn_activations + eps), axis=(-2,-1))
#         log_normalize_const = tf.reshape(frob_norm + ent_sinkhorn_acti, (-1, 1))
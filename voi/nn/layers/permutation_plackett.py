from voi.nn.wrappers.layer import Layer
from voi.nn.base.block import Block
from voi.nn.input import AttentionInput
from voi.common_enum import *
from voi.misc import custom_gather
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.optimize import linear_sum_assignment
import numpy as np

#@tf.function
def deterministic_NeuralSort(logs, mask):
    """
    logs: input elements to be sorted. Shape: batch_size x n x 1
    mask: Shape: batch_size x n x 1 
    
    Return: 
    P_sort(logs), shape batch_size x n x n
    Note that P_sort^hat is different from the one in the original
    paper. It has all invalid positions set to negative inf. In the
    valid positions, it is not a softmax matrix because we cannot feed the 
    softmax matrix to calculate the autoregressive decoder loss. 
    Instead, the matrix returned by this function is manually processed by find_permu
    to obtain the hard permutation matrix through argmax.
    """
    # The reason we can take exp() here is that find_permu
    # uses argmax to obtain the hard permutation matrix. Thus working in logs or
    # exp(logs) doesn't matter theoretically. Working in exp(logs) 
    # has better numerical stability.
    s = tf.math.exp(logs) * mask
    n = tf.shape(s)[1]
    one = tf.ones((n, 1), dtype = tf.float32)
    A_s = tf.math.abs(s - tf.transpose(s, [0, 2, 1]))
    B = tf.matmul(A_s, tf.matmul(one, tf.transpose(one)))
    scaling = tf.cast(n + 1 - 2*(tf.range(n) + 1), dtype = tf.float32)
    C = tf.matmul(s, tf.expand_dims(scaling, 0))
    P_max = tf.transpose(C - B, perm=[0, 2, 1])
    P_hat = P_max
    P_hat = P_hat - (1 - tf.transpose(mask, (0,2,1))) * 1e9
    return P_hat        

# This requires wrapping with tf.py_function, so multi-gpu training is slow
def find_permu(matrix_batch):
    """
    Find the permutation matrix greedily based on scores.
    """
    sol = np.zeros((matrix_batch.shape[0], matrix_batch.shape[1]), dtype=np.int32)
    flag = np.zeros_like(sol)
    rangeb = np.arange(matrix_batch.shape[0])
    for j in range(matrix_batch.shape[1]):
        tmp = matrix_batch[:,j,:]
        tmp = tmp - flag * 1e9
        idx = np.argmax(tmp, axis=1)
        sol[:, j] = idx
        flag[rangeb, idx] = 1
    return sol
    
class PermutationPlackettLayer(Layer):

    def __init__(self,
                 hidden_size,
                 temperature=1.0,
                 use_gumbel_noise=True,
                 **kwargs):
        """Creates the last layer of Permutation Transformer by sampling permutations
        through Plackett-Luce distribution
        Arguments:
        hidden_size: int
            the number of units in the hidden variables used
            in each multi head attention layer
        temperature: float, UNUSED
            a positive number to divide the permutation logits by
        use_gumbel_noise: bool, UNUSED
            whether to apply gumbel noise to the output of PermutationLayer"""
        
        super(PermutationPlackettLayer, self).__init__()

        # the core attention and processing variables
        self.block0 = Block(hidden_size // 2,
                            1,
                            **kwargs)
        
        self.hidden_size = hidden_size
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
                the logits for the Plackett-Luce distribution, 
                i.e. X = q(y, x) and here X is 1d, after masking out <start> <end> <pad>
                the shape is [batch * action_refinement, seq_length]
            kl: 
                KL((X+eps)/self.temperature || eps/temp_prior), 
                where X is the logits above,
                      self.temperature = 1.0,
                      temp_prior = 1.0 and eps ~ gumbel noise;
                the shape is [batch * action_refinement, 1]
                This assumes that the log score prior is 0.
                Since Plackett-Luce doesn't really have a prior,
                this KL term is only used to regularize the logits.
            log_nominator: 
                log of nominator of q(P|y, x);
                the shape is [batch * action_refinement, 1]
            log_normalize_const: 
                log of denominator of q(P|y, x);
                the shape is [batch * action_refinement, 1]
        """
        
        queries = inputs[QUERIES]
        queries_mask = inputs[QUERIES_MASK]   
        pretrain_done = inputs[PRETRAIN_DONE]
        action_refinement = inputs[ACTION_REFINEMENT]
        
        """
        The following implementation is kind of weird
        since we can't work in log-space, as the actual
        non-log scores are required in deterministic_NeuralSort.
        Unfortunately, working in log-space will cause the 
        exp(logs) to be inf and crash the program.
        """        
        shape = tf.shape(queries)
        # get log(s), where s > 0 is the score tensor
        activations = tf.maximum(tf.math.softplus(self.block0(queries, **kwargs)), 1e-5)
        activations = tf.math.log(activations)
        # prevent two activations from being identical,
        # especially the two activations that are originally less than 1e-5
        noise = tf.random.uniform(shape=tf.shape(activations), maxval=1e-5)
        activations += noise
        
        # repeat the tensor so that we can sample action_refinement permutations
        # per training data
        activations = tf.repeat(activations, action_refinement, axis=0) # (batch, len, 1)
        sqz_activations = tf.squeeze(activations, axis=2)
        
        queries_mask = tf.repeat(tf.expand_dims(queries_mask, 1), action_refinement, axis=0)
        valid_activation_mask = tf.cast(queries_mask, tf.float32) # (batch, 1, len)
        n_valid = tf.reduce_sum(valid_activation_mask, axis=-1)
        onedim_va_mask = tf.transpose(valid_activation_mask, [0,2,1])
        sqz_onedim_va_mask = tf.squeeze(onedim_va_mask, axis=2)
        # set log activations in invalid positions (start, end, pad) to be negative inf
        masked_activations = tf.where(onedim_va_mask > 0, activations,
                                      tf.ones_like(activations) * (-1e6))                                                   
        twodim_va_mask = tf.matmul(valid_activation_mask, valid_activation_mask,
                                    transpose_a = True) # (batch, len, len)
        
        # add Gumbel noise to log(s), and perform NeuralSort
        g = tfp.distributions.Gumbel(
                loc=tf.zeros_like(activations), 
                scale=tf.ones_like(activations))        
        perturb_acti = masked_activations + g.sample()
        perturb_acti = deterministic_NeuralSort(perturb_acti, onedim_va_mask)
        id_permu = tf.cast(tf.range(shape[1])[tf.newaxis, :], tf.int32)
        chosen_idx = tf.py_function(func=find_permu, inp=[perturb_acti], Tout=tf.int32) # 2D
        chosen_idx.set_shape(tf.TensorShape([None, None]))   
        # chosen_idx's first element is not the <start> token due to us masking out its
        # activations. Thus to obtain a valid permutation, we need to manually append a 
        # zero to the beginning.
        chosen_idx = chosen_idx[:, :-1]
        chosen_idx = tf.concat(
            [tf.zeros([tf.shape(chosen_idx)[0], 1], 
                      dtype=tf.int32
                     ), 
             chosen_idx],
            axis=-1
        )
        onedim_sample_permu = tf.where(sqz_onedim_va_mask > 0, chosen_idx, id_permu)
        sample_permu = tf.one_hot(onedim_sample_permu, depth=shape[1], axis=-1)  
        
        # Calculate the log probability of sampled permutations
        exp_actis = custom_gather(tf.squeeze(masked_activations, 2), onedim_sample_permu)
        exp_actis = tf.math.exp(exp_actis)
        reverse_cumsum_exp_actis = tf.math.cumsum(exp_actis[:, ::-1], axis=-1)[:, ::-1]
        eps = 1e-20
        log_nominator = tf.math.log(exp_actis + eps)
        log_nominator = log_nominator * sqz_onedim_va_mask
        log_nominator = tf.reduce_sum(log_nominator, axis=-1, keepdims=True)
        log_normalize_const = tf.math.log(reverse_cumsum_exp_actis + eps)
        log_normalize_const = log_normalize_const * sqz_onedim_va_mask
        log_normalize_const = tf.reduce_sum(log_normalize_const, axis=-1, keepdims=True)
        
        # calculate kl divergence KL((X+eps)/self.temperature || eps/temp_prior), 
        # as in Gumbel-Sinkhorn paper except that X is in 1d,
        # where eps ~ gumbel noise; self.temperature = 1.0 and temp_prior = 1.0
        kl_term1 = (
            n_valid 
            * (tf.math.log(self.temperature) 
               - 1.0
               + np.euler_gamma * (1.0 / self.temperature - 1.0)
              )
        )
        s1 = (1.0 
              / self.temperature
             * tf.reshape(
                 tf.reduce_sum(sqz_activations * sqz_onedim_va_mask, axis=-1), 
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
                        sqz_activations * sqz_onedim_va_mask, 
                        -20.0 * self.temperature)
                ),
                axis=-1
            ), 
            (-1,1)
        ) 
        s2 = s2 - (tf.cast(shape[1], tf.float32) - n_valid)
        kl = kl_term1 + s1 + s2 * tf.math.exp(tf.math.lgamma(1.0 + 1.0 / self.temperature))
        
        tf.print("kl, s1, s2", tf.squeeze(kl), tf.squeeze(s1), tf.squeeze(s2), summarize=-1)         
        
        return [sample_permu, tf.squeeze(masked_activations, 2), kl, 
                log_nominator, log_normalize_const]
    
    
    
# Debugging    
#         for i in range(tf.shape(onedim_sample_permu)[0]):
#             if tf.reduce_sum(onedim_sample_permu, axis=-1)[i] != 231:
#                 tf.print("nan activations", sqz_activations[i], summarize=-1)
#                 tf.print("nan perturb acti", perturb_acti[i], summarize=-1)
#                 tf.print("nan chosen idx", chosen_idx[i], summarize=-1)
#                 tf.print("nan mask", sqz_onedim_va_mask[i], summarize=-1)
#                 tf.print("nan matching", matching2d(perturb_acti)[i], summarize=-1)

#         tf.print("sample permu [:3]", sample_permu[:3], summarize=-1)
#         for idx in range(3):
#             locs = tf.where(sample_permu[idx] == 1.0)
#             d2 = tf.shape(locs)[1]
#             locs = tf.reshape(locs, [locs[-1,0]+1, d2])
#             tf.print("Sampled 3 permutations:",
#                      locs[:, -1], "\n", summarize=-1)      
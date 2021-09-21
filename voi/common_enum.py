# ids of special tokens
PAD_ID = 0 # pad_id should always be 0 by convention
UNK_ID = 1
START_ID = 2
END_ID = 3

"""
Unfortunately, due to issues in tensorflow's handling of python Dataclass
when running in static graph mode, we used a list of tensors to 
represent inputs and permu_inputs, and we used indices to access the corresponding tensors.

List indices are shown below:

inputs (inputs to autoregressive decoder):
    [0] = queries, target sequence
    [1] = values, source sequence
    [2] = queries_mask, mask for target sequence
    [3] = values_mask, mask for source sequence
    [4] = ids, ground truth ids of target tokens to decode at each time step
    [5] = permutation, the ordering of target tokens which is viewed as the 
          ground truth by the decoder
    [6] = absolute_positions
    [7] = relative_positions, only used for the decoder part of autoregressive decoder
    [8] = pointer_labels
    [9] = logits_labels
    [10] = partial_pos
    [11] = pointer_probs
    [12] = log_probs
    [13] = object_detections, used for captioning tasks only
    [14] = object_features, used for captioning tasks only
    [15] = object_boxes, used for captioning tasks only
permu_inputs (inputs to Permutation Transformer):
    [0] = queries, target sequence 
    [1] = values, source sequence
    [2] = queries_mask, mask for target sequence
    [3] = values_mask, mask for source sequence
    [4] = pretrain_done, True if decoder has finished pretraining with
          uniformly sampled orderings
    [5] = action_refinement, number of permutations to sample per training data
    [6] = None, unused
    [7] = relative_positions, only used for the decoder part of Permutation Transformer if pt_relative_embedding=True 
    [8] = tags, parts of speech tags of the target sequence
    [9] = None, unused
    [10] = activations, values assigned from the returned results of order(permu_inputs)
    [11] = kl, values assigned from the returned results of order(permu_inputs)
    [12] = log of permutation probabilities, values assigned from the returned results of order(permu_inputs)
    [13] = object_detections, used for captioning tasks only
    [14] = object_features, used for captioning tasks only
    [15] = object_boxes, used for captioning tasks only

For inputs to the autoregressive decoder, "queries" have size [batch_size, len - 1]
due to teacher forcing. "queries_mask" have size [batch_size, len - 1], which equals 1 if 
the token is not <pad> (i.e. id is not 0) and 0 otherwise.

For inputs to the Permutation Transformer, "queries" have size [batch_size, len]
since it uses the entire target ground truth sequence to generate permutations
for the decoder to learn. 
"queries_mask" have size [batch_size, len], which equals 1 if 
the token is not in [<start>, <end>, <pad>] and 0 otherwise.
This is because the generated permutation only permute the tokens
other than [<start>, <end>, <pad>]. Tokens in [<start>, <end>, <pad>]
are always fixed in either the beginning or the end of sentences,
and they should not be factored in the loss calculation.
"""

"""
token_indicators: 1 if token is not <pad>; 0 otherwise
e.g. 
<start> A dog . <end> <pad> <pad>
   1    1  1  1   1     0     0
"""

# autoregressive decoder 
LEN_INPUTS = 16

QUERIES = 0
VALUES = 1
QUERIES_MASK = 2
VALUES_MASK = 3
IDS = 4
PERMUTATION = 5
ABSOLUTE_POSITIONS = 6
RELATIVE_POSITIONS = 7
POINTER_LABELS = 8
LOGITS_LABELS = 9
PARTIAL_POS = 10
POINTER_PROBS = 11
LOG_PROBS = 12
OBJECT_DETECTIONS = 13
OBJECT_FEATURES = 14
OBJECT_BOXES = 15

# Permutation Transformer, if different
PRETRAIN_DONE = 4
ACTION_REFINEMENT = 5
TAGS = 8
ACTIVATIONS = 10
KL = 11
LOG_PERMU_PROBS = 12


"""
Example usage:

inputs = [None for _ in range(LEN_INPUTS)]
inputs[QUERIES] = queries
inputs[VALUES] = values
inputs[QUERIES_MASK] = queries_mask
inputs[VALUES_MASK] = values_mask
inputs[IDS] = ids
inputs[PERMUTATION] = permutation
inputs[ABSOLUTE_POSITIONS] = absolute_positions
inputs[RELATIVE_POSITIONS] = relative_positions
inputs[POINTER_LABELS] = pointer_labels
inputs[LOGITS_LABELS] = logits_labels
inputs[PARTIAL_POS] = partial_pos
inputs[POINTER_PROBS] = pointer_probs
inputs[LOG_PROBS] = log_probs    
inputs[OBJECT_DETECTIONS] = object_detections
inputs[OBJECT_FEATURES] = object_features
inputs[OBJECT_BOXES] = object_boxes

queries = inputs[QUERIES]
values = inputs[VALUES]
queries_mask = inputs[QUERIES_MASK]
values_mask = inputs[VALUES_MASK]
ids = inputs[IDS]
permutation = inputs[PERMUTATION]
absolute_positions = inputs[ABSOLUTE_POSITIONS]
relative_positions = inputs[RELATIVE_POSITIONS]
pointer_labels = inputs[POINTER_LABELS]
logits_labels = inputs[LOGITS_LABELS]
partial_pos = inputs[PARTIAL_POS]
pointer_probs = inputs[POINTER_PROBS]
log_probs = inputs[LOG_PROBS]
object_detections = inputs[OBJECT_DETECTIONS]
object_features = inputs[OBJECT_FEATURES]
object_boxes = inputs[OBJECT_BOXES]
"""
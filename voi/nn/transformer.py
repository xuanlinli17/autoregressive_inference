from voi.nn.wrappers.sequential import Sequential
from voi.nn.layers.encoder_layer import EncoderLayer
from voi.nn.layers.encoder_with_relative_position_layer import EncoderWithRelativePositionLayer
from voi.nn.layers.decoder_layer import DecoderLayer
from voi.nn.layers.decoder_with_relative_position_layer import DecoderWithRelativePositionLayer
from voi.nn.features.discrete_feature import DiscreteFeature
from voi.nn.features.region_feature import RegionFeature
from voi.nn.variables.logits import Logits
from voi.nn.variables.pointer_after_logits import PointerAfterLogits
import tensorflow as tf


class Transformer(Sequential):

    def __init__(self,
                 num_tgt_embeddings,
                 hidden_size,
                 heads,
                 num_layers,
                 src_embedding,
                 tgt_embedding,
                 queries_dropout=0.,
                 keys_dropout=0.,
                 values_dropout=0.,
                 label_smoothing=0.,
                 causal=True,
                 first_layer='region',
                 final_layer='logits',
                 sinusoid_pos_emb=False,
                 dataset='captioning',
                 **kwargs):
        """Creates a Transformer Keras model for processing sequences
        and uses the tf.layers.Sequential as backend

        Arguments:

        num_tgt_embeddings: int
            the number of elements in the target vocabulary which
            input sequences contain elements of
        hidden_size: int
            the number of units in the hidden variables used
            in each multi head attention layer
        heads: int
            the number of heads in each multi head attention layer
            a good default is 4 or 8
        num_layers: int
            the number of variables in the encoder and the decoder modules
            each layer consists of attention residual connections
        src_embedding: tf.keras.layers.Embedding
            the source embedding shared between the decoder
            and the Permutation Transformer
            in image captioning, this is the source detection
            in translation, this is the source vocab embedding
        tgt_embedding: tf.keras.layers.Embedding
            the target embedding shared between the decoder
            and the Permutation Transformer  
            in image captioning, this is the target caption
            in translation, this is the target vocab embedding
        queries_dropout: float
            the ratio of units to drop during training to the
            number of units in each attention layer
        keys_dropout: float
            the ratio of units to drop during training to the
            number of units in each attention layer
        values_dropout: float
            the ratio of units to drop during training to the
            number of units in each attention layer
        label_smoothing: float
            label smoothing coefficient
        causal: bool
            specifies is the transformer should decoding using
            a causal mask to preserve the auto regressive property
        first_layer: class
            specifies the class to use for the first layer in the transformer
            defaults to WordFeature if not specified
        final_layer: class
            specifies the class to use for the final layer in the transformer
            defaults to Logits if not specified
        sinusoid_pos_emb: bool
            whether to add positional embedding to the decoder to let it know
            its own generation ordering
        dataset: str
            type of dataset"""

        # TODO: Sequential does not technically support nested inputs
        layers = []
        super(Transformer, self).__init__(layers)

        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
#         # initialize embedding immediately since we need the entire
#         # embedding matrix
#         self.queries_embedding(tf.constant([1]))
#         self.values_embedding(tf.constant([1]))
        
        # the first layer in the transformer depends on the data modality
        # for image captioning using RCNN features select 'region'
        if first_layer == 'discrete':
            layers.extend([DiscreteFeature(
                hidden_size, 
                self.src_embedding, self.tgt_embedding, mode='decoder', 
                sinusoid_pos_emb=sinusoid_pos_emb, **kwargs)])
        if first_layer == 'region':
            layers.extend([RegionFeature(
                hidden_size,
                self.src_embedding, self.tgt_embedding, mode='decoder', 
                sinusoid_pos_emb=sinusoid_pos_emb, **kwargs)])

        # the encoder processes values and the decoder processes queries
        # thus we build the encoder first in tf.keras.sequential model
        # note that for captioning tasks, encoder doesn't have relative position
        # available
        if dataset == 'captioning' or sinusoid_pos_emb:
            layers.extend([EncoderLayer(
                hidden_size, hidden_size * 4, heads,
                queries_dropout=queries_dropout,
                keys_dropout=keys_dropout,
                values_dropout=values_dropout,
                causal=False, **kwargs) for _ in range(num_layers)])
        else:
            layers.extend([EncoderWithRelativePositionLayer(
                hidden_size, hidden_size * 4, heads,
                queries_dropout=queries_dropout,
                keys_dropout=keys_dropout,
                values_dropout=values_dropout,
                causal=False, num_pos=1, **kwargs) for _ in range(num_layers)])            

        # depending on the type of network possibly condition on position
        # build the decoder second in the stack
        cls = (DecoderWithRelativePositionLayer
               if final_layer == 'indigo' else DecoderLayer)
        layers.extend([cls(
            hidden_size, hidden_size * 4, heads,
            queries_dropout=queries_dropout,
            keys_dropout=keys_dropout,
            values_dropout=values_dropout,
            causal=causal, **kwargs) for _ in range(num_layers)])
        
        # the final layer in the transformer depends on the model purpose
        # to run Transformer-InDIGO select 'indigo'
        if final_layer == 'logits' or final_layer == 'indigo':
            layers.extend([Logits(hidden_size, self.tgt_embedding, label_smoothing, **kwargs)])
        if final_layer == 'indigo':
            layers.extend([PointerAfterLogits(
                hidden_size * 4, hidden_size, num_tgt_embeddings, self.tgt_embedding,
                causal=causal, **kwargs)])

        self.final_layer_obj = layers[-1]
        
        super(Transformer, self).__init__(layers)
        print("Autoregressive decoder layers", layers)

        self.num_tgt_embeddings = num_tgt_embeddings
        self.hidden_size = hidden_size
        self.heads = heads
        self.num_layers = num_layers
        self.queries_dropout = queries_dropout
        self.keys_dropout = keys_dropout
        self.values_dropout = values_dropout
        self.label_smoothing = label_smoothing
        self.causal = causal
        self.first_layer = first_layer
        self.final_layer = final_layer
        self.sinusoid_pos_emb = sinusoid_pos_emb
        self.dataset = dataset
        self.kwargs = kwargs
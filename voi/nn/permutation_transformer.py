from voi.nn.wrappers.sequential import Sequential
from voi.nn.layers.encoder_layer import EncoderLayer
from voi.nn.layers.encoder_with_relative_position_layer import EncoderWithRelativePositionLayer
from voi.nn.layers.encoder_with_relative_positional_attention_layer import EncoderWithRelativePositionalAttentionLayer
from voi.nn.layers.decoder_layer import DecoderLayer
from voi.nn.layers.decoder_with_relative_positional_attention_layer import DecoderWithRelativePositionalAttentionLayer
from voi.nn.layers.decoder_with_relative_position_layer import DecoderWithRelativePositionLayer
from voi.nn.layers.permutation_sinkhorn import PermutationSinkhornLayer
from voi.nn.layers.permutation_plackett import PermutationPlackettLayer
from voi.nn.features.discrete_feature import DiscreteFeature
from voi.nn.features.region_feature import RegionFeature
from voi.nn.features.pt_discrete_with_tags_feature import PtDiscreteWithTagsFeature


class PermutationTransformer(Sequential):

    def __init__(self,
                 hidden_size,
                 heads,
                 num_layers,
                 src_embedding,
                 tgt_embedding,  
                 tags_embedding=None,
                 tags_embedding_size=256,
                 share_encoder_model=None,
                 queries_dropout=0.,
                 keys_dropout=0.,
                 values_dropout=0.,
                 first_layer='region',
                 pg_final_layer='plackett',
                 pt_positional_attention=True,
                 pt_relative_embedding=False,
                 pt_special_encoder_block='none',
                 temperature=1.,
                 dataset='captioning',
                 hungarian_op_path='',
                 **kwargs):
        """Creates a Transformer Keras model for processing sequences
        and uses the tf.layers.Sequential as backend

        Arguments:
        
        hidden_size: int
            the number of units in the hidden variables used
            in each multi head attention layer
        heads: int
            the number of heads in each multi head attention layer
            a good default is 4 or 8
        num_layers: int
            the number of variables in the encoder and the decoder modules
            each layer consists of attention residual connections
        tags_embedding: tf.keras.layers.Embedding
            the parts of speech tags embedding; None if not used
        tags_embedding_size: int
            the embedding size for parts of speech tags, if tags_embedding is used
        share_encoder_model: tf.keras.model or None
            if share_encoder_model is not None, then the transformer encoder is shared
            between the Permutation Transformer and the autoregressive decoder transformer
        queries_embedding: tf.keras.layers.Embedding
            the queries embedding shared between the decoder
            and the Permutation Transformer
            in image captioning, this is the source detection
            in translation, this is the source vocab embedding
        values_embedding: tf.keras.layers.Embedding
            the values embedding shared between the decoder
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
        first_layer: class
            specifies the class to use for the first layer in the transformer
            defaults to WordFeature if not specified
        pg_final_layer: class
            if policy gradient is 'without_bvn',
            specifies the class to use for the final layer in the transformer
            defaults to 'plackett' if not specified     
        pt_positional_attention: bool
            whether to use positional attention
        pt_relative_embedding: bool
            whether to use relative embedding instead of positional embedding
        pt_special_encoder_block: str
            Use a special encoder block type for Permutation Transformer, 
            overriding all other pt-encoder related settings.        
        temperature: float
            a positive number to divide the permutation logits by prior
            to applying sinkhorn normalization
        dataset: str
            type of dataset
        hungarian_op_path: str
            the path to the cpu / gpu op of hungarian algorithm (for 
            obtaining hard permutation matrices from soft permutation
            matrices) """

        # TODO: Sequential does not technically support nested inputs
        layers = []
        super(PermutationTransformer, self).__init__(layers)

        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.tags_embedding = tags_embedding
        self.tags_embedding_size = tags_embedding_size
        
        # the first layer in the transformer depends on the data modality
        # for image captioning using RCNN features select 'region'
        print("PT sinusoid embedding", not pt_relative_embedding and not pt_positional_attention)
        sinusoid_pos_emb = not pt_relative_embedding and not pt_positional_attention
        if first_layer == 'discrete':
            if self.tags_embedding is None:
                layers.extend([DiscreteFeature(
                    hidden_size, 
                    self.src_embedding, self.tgt_embedding, mode='pt', 
                    sinusoid_pos_emb=sinusoid_pos_emb,
                    **kwargs)])
            else:
                layers.extend([PtDiscreteWithTagsFeature(
                    hidden_size, tags_embedding_size,
                    self.src_embedding, self.tgt_embedding, self.tags_embedding,
                    sinusoid_pos_emb=sinusoid_pos_emb,
                    **kwargs)])
        if first_layer == 'region':
            layers.extend([RegionFeature(
                hidden_size,
                self.src_embedding, self.tgt_embedding, mode='pt', 
                sinusoid_pos_emb=sinusoid_pos_emb,
                **kwargs)])
            
        # the encoder processes values and the decoder processes queries
        # thus we build the encoder first in tf.keras.sequential model
        
        # since we shared the encoder weights with those of the autoregressive decoder before finetuning,
        # we need to use the same encoder layer class as that of autoregressive decoder
        if share_encoder_model is not None:
            for l in share_encoder_model.layers:
                if isinstance(l, (EncoderLayer, EncoderWithRelativePositionLayer, EncoderWithRelativePositionalAttentionLayer)):
                    layers.append(l)
        else:
            if pt_special_encoder_block != 'none':
                pt_special_encoder_block = eval(pt_special_encoder_block)
            else:
                pt_special_encoder_block = None
            if (pt_special_encoder_block == EncoderLayer 
                or (pt_special_encoder_block != EncoderLayer 
                    and (
                        dataset == 'captioning' 
                        or not (pt_relative_embedding or pt_positional_attention)
                    )
                   )
               ):
                layers.extend([EncoderLayer(
                    hidden_size, hidden_size * 4, heads,
                    queries_dropout=queries_dropout,
                    keys_dropout=keys_dropout,
                    values_dropout=values_dropout,
                    causal=False, **kwargs) for _ in range(num_layers)])
            elif (pt_special_encoder_block == EncoderWithRelativePositionLayer
                  or (pt_special_encoder_block != EncoderWithRelativePositionLayer
                      and pt_relative_embedding)
                 ):
                layers.extend([EncoderWithRelativePositionLayer(
                    hidden_size, hidden_size * 4, heads,
                    queries_dropout=queries_dropout,
                    keys_dropout=keys_dropout,
                    values_dropout=values_dropout,
                    causal=False, num_pos=1, **kwargs) for _ in range(num_layers)])
            elif (pt_special_encoder_block == EncoderWithRelativePositionalAttentionLayer
                  or (pt_special_encoder_block != EncoderWithRelativePositionalAttentionLayer
                      and pt_positional_attention)
                 ):
                layers.extend([EncoderWithRelativePositionalAttentionLayer(
                    hidden_size, hidden_size * 4, heads,
                    queries_dropout=queries_dropout,
                    keys_dropout=keys_dropout,
                    values_dropout=values_dropout,
                    causal=False, **kwargs) for _ in range(num_layers)])            

        # depending on the type of network possibly condition on position
        # build the decoder second in the stack
        if pt_positional_attention:
            layers.extend([DecoderWithRelativePositionalAttentionLayer(
                hidden_size, hidden_size * 4, heads,
                queries_dropout=queries_dropout,
                keys_dropout=keys_dropout,
                values_dropout=values_dropout,
                causal=False, **kwargs) for _ in range(num_layers)])
        elif pt_relative_embedding:
            layers.extend([DecoderWithRelativePositionLayer(
                hidden_size, hidden_size * 4, heads,
                queries_dropout=queries_dropout,
                keys_dropout=keys_dropout,
                values_dropout=values_dropout,
                causal=False, **kwargs) for _ in range(num_layers)])            
        else:
            layers.extend([DecoderLayer(
                hidden_size, hidden_size * 4, heads,
                queries_dropout=queries_dropout,
                keys_dropout=keys_dropout,
                values_dropout=values_dropout,
                causal=False, **kwargs) for _ in range(num_layers)])            

        # the final layer in the transformer depends on the model purpose
        # to run Transformer-InDIGO select 'indigo'
        if pg_final_layer == 'sinkhorn':
            layers.extend([PermutationSinkhornLayer(
                hidden_size * 4, heads,
                queries_dropout=queries_dropout,
                keys_dropout=keys_dropout, 
                hungarian_op_path=hungarian_op_path, 
                temperature = temperature, **kwargs)])
        elif pg_final_layer == 'plackett':
            layers.extend([PermutationPlackettLayer(
                hidden_size * 4,
                temperature = temperature, **kwargs)])

        super(PermutationTransformer, self).__init__(layers)
        print("Permutation transformer layers", layers)
        self.last_layer = layers[-1]
        
        self.hidden_size = hidden_size
        self.heads = heads
        self.num_layers = num_layers
        self.queries_dropout = queries_dropout
        self.keys_dropout = keys_dropout
        self.values_dropout = values_dropout
        self.first_layer = first_layer
        self.pg_final_layer = pg_final_layer
        self.pt_positional_attention = pt_positional_attention
        self.pt_relative_embedding = pt_relative_embedding
        self.temperature = temperature
        self.dataset = dataset
        self.kwargs = kwargs
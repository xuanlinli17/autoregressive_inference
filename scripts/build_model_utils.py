import tensorflow as tf
from voi.nn.transformer import Transformer
from voi.nn.permutation_transformer import PermutationTransformer
from voi.process.captions import Vocabulary
from voi.common_enum import *

def build_model(args, return_pt=True, no_dropout=False):    
    """
    Builds the models, vocabularies, and parallel strategies based on input arguments
    
    Arguments:
        args: arguments parsed thru ArgParse
        return_pt: whether to construct and return the Permutation Transformer, along with
            the parts-of-speech vocab used to train the Permutation Transformer (if args.tags_vocab_file != '')
        no_dropout: whether to eliminate dropout during model construction
    
    Returns:
        model: the autoregressive decoder model (tf.keras.Model)
        order: the Permutation Transformer (tf.keras.Model), if return_pt == True and args.order == 'soft';
            otherwise returns None
        vocab: the vocabulary file for the target and source sequences
        tags_vocab: the vocabulary file for parts of speech, 
            if return_pt == True and args.order == 'soft' and args.tags_vocab_file != ''
            (i.e. if we provide parts of speech information to train the Permutation Transformer);
            otherwise returns None
        strategy: tf.distribute.Strategy, tensorflow strategy for training in multi-gpu settings
    """
    assert not (args.pt_positional_attention and args.pt_relative_embedding)
    if args.dataset == 'captioning':
        assert args.first_layer == 'region'
    elif args.dataset in ['wmt', 'django']:
        assert args.first_layer == 'discrete'
        
    if args.finetune_decoder_transformer:
        # train decoder for infinite num of steps so that Permutation Transformer doesn't get trained
        args.alternate_training = [10000000, 1] 
    if args.alternate_training is not None:
        assert len(args.alternate_training) == 2
        assert args.order == 'soft'
    if args.order == 'soft':
        assert args.action_refinement >= 1
    else:
        assert args.action_refinement == 1
        assert not args.use_ppo
        
    assert args.warmup >= 0
    assert '.h5' == args.model_ckpt[-3:], "Please save the model in hdf5 format"
    
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

#     Different tf.distribute.Strategy
#     if args.parallel_strategy == 'nccl':
#         strategy = tf.distribute.MirroredStrategy(
#             cross_device_ops=tf.distribute.NcclAllReduce())
#     elif args.parallel_strategy == 'hierarchy':
#         strategy = tf.distribute.MirroredStrategy(
#             cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())       
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    with tf.io.gfile.GFile(args.vocab_file, "r") as f:
        vocab = Vocabulary([x.strip() for x in f.readlines()],
                           unknown_word="<unk>",
                           unknown_id=UNK_ID)

    with strategy.scope():
        
        def get_src_tgt_embedding():
            if args.dataset == 'captioning':
                return (
                    tf.keras.layers.Embedding(91, args.embedding_size),
                    tf.keras.layers.Embedding(vocab.size(), args.embedding_size)
                ) 
            elif args.dataset == 'wmt':
                emb = tf.keras.layers.Embedding(
                            vocab.size(), args.embedding_size)       
                return emb, emb
            elif args.dataset == 'django':
                emb = tf.keras.layers.Embedding(
                            vocab.size(), args.embedding_size)
                return emb, emb
            elif args.dataset == 'gigaword':
                emb = tf.keras.layers.Embedding(
                            vocab.size(), args.embedding_size)
                return emb, emb            
            
        model_src_embedding, model_tgt_embedding = get_src_tgt_embedding()  
        if args.share_embedding:
            pt_src_embedding = model_src_embedding
            pt_tgt_embedding = model_tgt_embedding
        else:
            pt_src_embedding, pt_tgt_embedding = get_src_tgt_embedding() 
        
        # The decoder autoregressive transformer's "call" function
        # is dummy by design and is only used for initializing the parameters.
        # Do not call 'model(inputs)' to get the results. 
        # Instead, use the model's loss function and the beam/greedy search function
        # to obtain results        
        model = Transformer(vocab.size(),
                            args.embedding_size,
                            args.heads,
                            args.num_layers,
                            model_src_embedding,
                            model_tgt_embedding,
                            queries_dropout=args.queries_dropout if not no_dropout else 0.0,
                            keys_dropout=args.keys_dropout if not no_dropout else 0.0,
                            values_dropout=args.values_dropout if not no_dropout else 0.0,
                            causal=True,
                            first_layer=args.first_layer,
                            final_layer=args.final_layer,
                            sinusoid_pos_emb=args.sinusoid_pos_embedding,
                            dataset=args.dataset,
                            label_smoothing=args.label_smoothing)

        tags_vocab = None
        tags_embedding = None
        order = None
        if args.order == 'soft' and return_pt:
            if args.tags_vocab_file != '':
                with tf.io.gfile.GFile(args.tags_vocab_file, "r") as f:
                    tags_vocab = Vocabulary([x.strip() for x in f.readlines()],
                        unknown_word="<unk>",
                        unknown_id=UNK_ID)        
                tags_embedding = tf.keras.layers.Embedding(
                    tags_vocab.size(),
                    args.tags_embedding_size)
            # The Permutation Transformer can be directly
            # called on the inputs to obtain sampled 
            # permutations and probabilities, 
            # i.e. through 'order(permu_inputs)'
            order = PermutationTransformer(args.embedding_size,
                                           args.heads,
                                           args.num_layers,
                                           pt_src_embedding,
                                           pt_tgt_embedding,
                                           tags_embedding=tags_embedding,
                                           tags_embedding_size=args.tags_embedding_size,
                                           share_encoder_model=model if args.share_encoder else None,
                                           queries_dropout=0.1 if not no_dropout else 0.0,
                                           keys_dropout=0.1 if not no_dropout else 0.0,
                                           values_dropout=0.1 if not no_dropout else 0.0,
                                           first_layer=args.first_layer,
                                           pg_final_layer=args.pt_pg_type,
                                           pt_positional_attention=args.pt_positional_attention,
                                           pt_relative_embedding=args.pt_relative_embedding,
                                           pt_special_encoder_block=args.pt_special_encoder_block,
                                           dataset=args.dataset,
                                           hungarian_op_path=args.hungarian_op_path)    
            
        
    if return_pt:
        return model, order, vocab, tags_vocab, strategy
    else:
        return model, vocab, strategy
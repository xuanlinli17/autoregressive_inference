import argparse
def add_common_arguments(parser):    
    """
    File settings
    """
    parser.add_argument(
        '--dataset', type=str, default='captioning', 
        choices=['captioning', 'wmt', 'django', 'gigaword'],
        help="""
        Type of dataset.
        'captioning' = coco, flickr;
        'wmt' = machine translation;
        'django' = NL2Code;
        'gigaword' = text summarization
        """)
    parser.add_argument(
        '--vocab_file', type=str, default='',
        help="""
        Vocabulary file. Currently implemented as shared between 
        Permutation Transformer and autoregressive decoder.""")
    parser.add_argument(
        '--hungarian_op_path', type=str, default='./hungarian.so',
        help="""
        If training or validating with Variational Order Inference,
        Path to the tensorflow hungarian custom op.
        If the path is invalid or the loading fails, then 
        tf.py_function with scipy.optimize.linear_sum_assignment
        is used, which is not efficient for multi-gpu training.
        """)
    
    """
    Training folder
    """
    parser.add_argument(
        '--train_folder', type=str, default='tfrecords',
        help="Path to training set tfrecords folder")  
    
    """
    Validation, distillation, or inspect-order specific
    """
    parser.add_argument(
        '--validate_folder', type=str, default='tfrecords',
        help="Path to validation / test set tfrecords folder")  
    parser.add_argument(
        '--validation_output_save_path', type=str, default='', 
        help="""Save path for ground truth & model output (ref_caps_list.txt and hyp_caps_list.txt). 
                These two files are aligned through line number.
                Useless if we are validating coco, which has multiple references, 
                where metrics are computed on the fly.""")        
    parser.add_argument(
        '--caption_ref_folder', type=str, default='captions',
        help="""
        For validation/test, the path to a folder that contains ground truth sentence files
        ready to be loaded from the disk, for captioning datasets only.""")    
    parser.add_argument(
        '--visualization_save_path', type=str, default='inspect_generation_order_stats.txt',
        help='Visualization save path for sentences and orderings, for visualization only.')
    parser.add_argument(
        '--distillation_save_path', type=str, default='wmt16_ro_en/',
        help='If distilling existing models using sequence-level distillation through distill.py, the save directory of the output files.')    
    parser.add_argument(
        '--tagger_file', type=str, default='tagger.pkl',
        help='Parts of speech tagger file for visualizing generation orders in captioning datasets.')
    parser.add_argument(
        '--beam_size', type=int, default=1,
        help="Beam size for visualization during training, or beam size during validation.")
    
    """
    Model saving and loading
    """
    parser.add_argument(
        '--model_ckpt', type=str, default='ckpt/nsds.h5',
        help="Model checkpoint saving and loading. Must end in .h5")
    parser.add_argument(
        '--save_interval', type=int, default=50000,
        help="""
        The model snapshot saving interval.
        The snapshot is named by the # of iterations. 
        Disable if <=0 
        (model_name_ckpt.h5 etc still saved every 2000 iterations)
        """)
    
    """
    Model Configs
    """
    # Model basic config and sizes
    parser.add_argument(
        '--embedding_size', type=int, default=256,
        help="Size of token embedding.")    
    parser.add_argument(
        '--heads', type=int, default=4,
        help="Number of attention heads")
    parser.add_argument(
        '--num_layers', type=int, default=2,
        help="Number of attention layers")  
    parser.add_argument(
        '--first_layer', type=str,
        default='region', choices=['region', 'discrete', 'continuous'],
        help="""
        First layer of the decoder and Permutation Transformer.
        For captioning, choose "region".
        For other tasks, choose "discrete".
        """)
    parser.add_argument(
        '--final_layer', type=str,
        default='indigo', choices=['indigo', 'logits'],
        help="""
        Final layer of the autoregressive decoder.
        "indigo" is the Transformer INDIGO with 
        Logits and PointerAfterLogits layers and with
        both token and position losses. It generates
        the target sequences through insertion.
        "logits" is "indigo" without the PointerAfterLogits layer,
        i.e. there is no position loss in this case, and generation
        is left-to-right.
        For this argument, "indigo" covers all functionalities
        of "logits". In practice, when training with fixed orderings,
        select "indigo" for this argument and "l2r" "r2l" "rare" etc
        for the "orders" argument.
        """)    
    parser.add_argument(
        '--order', type=str,
        default='soft', 
        choices=['l2r', 'r2l', 'rare', 'common', 'test', 'soft', 'sao'],
        help="""
        The type of autoregressive ordering to train,
        either fixed ordering or VOI.
        l2r = left-to-right;
        r2l = right-to-left;
        rare = rare-first;
        common = common-first;
        soft = VOI;
        sao = searched adaptive order;
        test = debugging.
        
        In VOI, policy gradient is used to train the Permutation Transformer.
        The probability of "action" is calculated as follows:
            after applying the Hungarian algorithm on the soft 
            permutation from the Sinkhorn operation on the raw permutation
            transformer output X to obtain hard permutations P, 
            i.e. P = Hungarian(Sinkhorn((X + eps) / self.temperature)),
            where X = q(y, x) and q is the Permutation Transformer,
            the probabilities of hard 
            permutations are proportionally based on Gumbel-Matching distribution 
            i.e. exp(<X,P>_F), see https://arxiv.org/abs/1802.08665) 
        """)          
    # batch size, epochs, and learning rate
    parser.add_argument(
        '--batch_size', type=int, default=16,
        help="""
        Batch size for each training/validation iteration. 
        If training with Variational Order Inference, action_refinement permutations 
        are sampled per data, so the actual batch dimension
        has length batch_size * action_refinement.
        """)
    parser.add_argument(
        '--num_epochs', type=int, default=10,
        help="Number of training epochs.")    
    parser.add_argument(
        '--decoder_init_lr', type=float, default=0.0001,
        help="Decoder initial learning rate")      
    parser.add_argument(
        '--pt_init_lr', type=float, default=0.00001,
        help="Permutation Transformer initial learning rate") 
    parser.add_argument(
        '--lr_schedule', type=str, 
        default='linear', choices=['linear', 'constant'],
        help="Learning rate schedule, either linear decay or constant")
    parser.add_argument(
        '--warmup', type=int, default=0,
        help="Number of warmup iterations")   
    # dataset size if training a model, used to update lr and kl
    parser.add_argument(
        '--dataset_size', type=int, default=-1,
        help="""
        The size of training set. Used to calculate the total number of 
        training iterations, which is used to calculate the 
        learning rate at the current iteration along with current kl coefficient.
        If unspecified, then default sizes are used:
        'gigaword': 3803897;
        'wmt': 609324;
        'captioning': 591435;
        'django': 16000;
        """)               
    # variational order inference regularization
    parser.add_argument(
        '--kl_coeff', type=float, default=0.3,
        help="""
        Kl divergence coefficient beta
        where the loss is beta * KL((X+eps)/self.temperature || eps/temp_prior),
        where eps is Gumbel noise.
        This parameter is used for VOI training only.
        """)
    parser.add_argument(
        '--kl_log_linear', type=float, default=-1,
        help="""
        If this value > 0, decrease the log coefficient of kl
        linearly as training proceeds until the value given.
        This parameter is used for VOI training only.        .
        """)    
    # other regularization
    parser.add_argument(
        '--queries_dropout', type=float, default=0.1,
        help="Prob of queries dropout in attention")
    parser.add_argument(
        '--keys_dropout', type=float, default=0.1,
        help="Prob of keys dropout in attention")
    parser.add_argument(
        '--values_dropout', type=float, default=0.1,
        help="Prob of values dropout in attention")
    parser.add_argument(
        '--label_smoothing', type=float, default=0.0,
        help="Coefficient for label smoothing")
    # word and positional embedding
    parser.add_argument(
        '--share_embedding', action='store_true',
        help="""
        Whether to share the embedding between the 
        Permutation Transformer and the autoregressive decoder.
        This parameter is only used for VOI training.
        """)    
    parser.add_argument(
        '--pt_positional_attention', action='store_true',
        help="""
        Whether to use Transformer-XL relative position encoding
        for the Permutation Transformer. For VOI only.
        """)
    parser.add_argument(
        '--pt_relative_embedding', action='store_true',
        help="""
        Whether to use the relative (non-TransformerXL)-type
        positional embedding in the Permutation Transformer. For VOI only.
        """)
    parser.add_argument(
        '--sinusoid_pos_embedding', action='store_true',
        help="""
        Whether to use sinusoid position embedding for the
        autoregressive decoder.
        For Transformer-INDIGO this is set to False.
        """)
    parser.add_argument(
        '--embedding_align_coeff', type=float, default=0.0,
        help="""
        The coefficient of embedding alignment loss 
        between the Permutation Transformer and the decoder.
        This parameter is used for VOI training only.         
        """)
    # Variational Order Inference-specific
    parser.add_argument(
        '--pt_pg_type', type=str,
        default='sinkhorn', choices=['plackett', 'sinkhorn'],
        help="""
        Modeling q(.|x, y) as Gumbel-Sinkhorn distribution
        ("sinkhorn") or Plackett-Luce Distribution ("plackett")
        """)
    parser.add_argument(
        '--pt_special_encoder_block', type=str, default='none',
        choices=['none', 'EncoderLayer', 'EncoderWithRelativePositionLayer', 'EncoderWithRelativePositionalAttentionLayer'],
        help="""
        Use a special encoder block type for Permutation Transformer, overriding all other pt-encoder related settings.
        """
    )     
    parser.add_argument(
        '--share_encoder', action='store_true',
        help="""
        Whether to share the transformer encoder between 
        Permutation Transformer and the autoregressive decoder transformer.
        This parameter is only used for VOI training.
        """)
    parser.add_argument(
        '--action_refinement', type=int, default=1,
        help="""
        The number of actions (permutations, orderings) to sample
        per training data.
        The actual batch dimension
        has length batch_size * action_refinement.
        This parameter is used for VOI training only.
        """)
    parser.add_argument(
        '--use_ppo', action='store_true',
        help="""
        For policy gradient used to optimize Variational Order Inference, whether to use PPO.
        This parameter is used for VOI training only.
        """)
    parser.add_argument(
        '--finetune_decoder_transformer', action='store_true',
        help="""
        Whether to fix the Permutation Transformer in order to finetune the autoregressive decoder.
        This argument is for training VOI only.
        """)    
    parser.add_argument(
        '--alternate_training', nargs='+', type=int,
        help="""
        If two integers x and y are given, 
        then train the decoder and fix the Permutation Transformer for x iterations, 
        and then train the Permutation Transformer and fix the decoder for y iterations.
        Repeat this process until the end of training.
        """)
    parser.add_argument(
        '--decoder_training_scheme', type=str, default='all', choices=['best', 'all'],
        help="""
        Whether to train decoder with the best permutation ("best")
        or all sampled permutations from the Permutation Transformer ("all").
        This parameter is used for VOI training only.
        """)    
    parser.add_argument(
        '--decoder_pretrain', type=int, default=-1,
        help="""
        Number of batches for decoder pretraining using
        uniformly sampled permutations. Disabled if <0.
        After pretraining, orderings are sampled from
        the Permutation Transformer.
        """)     
    parser.add_argument(
        '--reward_std', action='store_true',
        help="""
        For policy gradient, whether to standardize the reward.
        This parameter is used for VOI training only.
        """)
    # whether to use parts of speech tags to train permutation transformer in Variational Order Inference;
    # we did not find this affect our learned orderings
    parser.add_argument(
        '--tags_vocab_file', default='', type=str,
        help="""
        If non empty string, then use parts of speech information to train the permutation
        transformer. The string specifies the path to the parts of speech vocab file.
        """)
    parser.add_argument(
        '--tags_embedding_size', default=-1, type=int,
        help="""
        The embedding vector size of parts of speech, if tags_vocab_file is not None.
        """)       
    
    """
    Miscellaneous
    """
    parser.add_argument(
        '--parallel_strategy', type=str, default='nccl', choices=['nccl', 'hierarchy'],
        help="""
        tf.distribute.MirroredStrategy options.
        'nccl' = NcclAllReduce;
        'hierarchy' = HierarchicalCopyAllReduce
        """)
    parser.add_argument(
        '--eval_frequency', type=int, default=500,
        help="During training, print out example generated sequences every eval_frequency timesteps"
    )
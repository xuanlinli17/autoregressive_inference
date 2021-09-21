import tensorflow as tf
# tf.config.threading.set_intra_op_parallelism_threads(1)
# tf.config.threading.set_inter_op_parallelism_threads(1)
from voi.core.train import train_dataset
from voi.nn.transformer import Transformer
from voi.nn.permutation_transformer import PermutationTransformer
from voi.process.captions import Vocabulary
import argparse
from common_argparse import add_common_arguments
from build_model_utils import build_model

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_arguments(parser)
    args = parser.parse_args()
    print(args)  
        
    #tf.profiler.experimental.server.start(6009)

    model, order, vocab, tags_vocab, strategy = build_model(args, return_pt=True, no_dropout=False)

    train_dataset(args.train_folder,
                  args.batch_size,
                  args.beam_size,
                  args.num_epochs,
                  model,
                  args.model_ckpt,
                  args.save_interval,
                  order if args.order == 'soft' else args.order,
                  vocab,
                  tags_vocab,
                  strategy,
                  args.dataset,
                  args.dataset_size,
                  args.reward_std,
                  args.pt_pg_type,
                  args.decoder_pretrain,
                  args.decoder_init_lr,
                  args.pt_init_lr,
                  args.lr_schedule,
                  args.warmup,
                  args.kl_coeff,
                  args.kl_log_linear,
                  args.action_refinement,
                  args.alternate_training,
                  args.use_ppo,
                  args.embedding_align_coeff,
                  args.decoder_training_scheme,
                  args.finetune_decoder_transformer,
                  args.eval_frequency)
